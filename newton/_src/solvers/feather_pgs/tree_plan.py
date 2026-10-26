# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One host plan of a joint forest: every traversal structure the tree scans and the factor read.

:meth:`TreePlan.build` takes a forest as each node's parent (``-1`` at a root)
and each node's coordinate count, and derives, with vectorized passes whose
count grows with the depth and not with the node count (so it serves a
template tree and a whole model alike):

* the topology: roots, children (CSR, descending index: the order of the
  serial leaf-to-root accumulation), depth and depth levels (CSR, ascending
  index), each node's root, subtree sizes;
* the pointer-jumping table (``jump[r, i]``, the ``2**r``-th ancestor);
* the depth-first order (children ascending) with each position's subtree end,
  so a subtree is the interval ``[position, end)``;
* the heavy-light decomposition: heavy child, chains, light children and the
  packing of padded chains into warps;
* ancestor masks over nodes and over each tree's coordinates;
* the fill-free factor pattern of the joint-space inertia
  (:meth:`TreePlan.factor`) and its row support classes
  (:meth:`TreePlan.support_classes`).

Plans of a template tree (at most 64 nodes and coordinates) also carry the
schedule data the native scans bake in; :func:`local_parents` maps a model's
global ancestors to template-local parents.

The host index operation below (``_group_by``) has the semantics of the host
index toolkit's ``group_by``; it moves there when the toolkit lands.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

__all__ = ["ChainPacking", "FactorPattern", "SupportClasses", "TreePlan", "local_parents"]


# ---------------------------------------------------------------------------------------------------------------------
# Host index operations (the host index toolkit's semantics)
# ---------------------------------------------------------------------------------------------------------------------


def _group_by(keys: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(offsets, order)``: ``order[offsets[k]:offsets[k + 1]]`` lists the positions holding key ``k``.

    Positions keep ascending order within a key (a stable grouping).
    """
    keys = np.asarray(keys, dtype=np.int64)
    order = np.argsort(keys, kind="stable")
    offsets = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(np.bincount(keys, minlength=count), out=offsets[1:])
    return offsets, order


def local_parents(ancestor: np.ndarray, starts: np.ndarray, count: int) -> np.ndarray:
    """Return template-local parents of ``count`` consecutive joints from each of ``starts``.

    ``parents[a, i]`` is the parent of joint ``starts[a] + i`` as an offset from
    ``starts[a]``, or ``-1`` at a root. An ancestor outside the range is not part
    of the group's trees and raises.
    """
    ancestor = np.asarray(ancestor, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64).reshape(-1, 1)
    joints = starts + np.arange(count)
    parents = ancestor[joints]
    inside = (parents >= starts) & (parents < starts + count)
    if np.any((parents >= 0) & ~inside):
        raise ValueError("A group joint's ancestor lies outside the group's joint range")
    return np.where(inside, parents - starts, -1).astype(np.int32)


# ---------------------------------------------------------------------------------------------------------------------
# Derived structures
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ChainPacking:
    """Heavy chains padded to powers of two and packed first-fit into warps, lane by lane.

    ``meta[lane]`` is ``(node, chain start lane, chain length, light level,
    anchor)`` with ``node = -1`` on an idle lane; the anchor is the parent of
    the chain's head (``-1`` for a root chain).
    """

    meta: np.ndarray
    """Per lane: node, start lane, length, light level, anchor (shape ``[32 * warps, 5]``)."""
    levels: int
    """Light levels (the scans' anchor stages)."""
    stage_width: np.ndarray
    """Widest padded chain of each light level."""
    width: int
    """Widest padded chain."""
    warps: int
    """Warps the packing spans."""

    @property
    def lanes(self) -> int:
        """Lanes spanned: the last occupied warp's end."""
        occupied = np.flatnonzero(self.meta[:, 0] >= 0)
        return 32 * (int(occupied[-1]) // 32 + 1) if occupied.size else 32


@dataclass(frozen=True, eq=False)
class FactorPattern:
    """The fill-free lower pattern of a tree's joint-space inertia, in leaf-first factor coordinates.

    Factor coordinate ``i`` names physical coordinate ``permutation[i]``: the
    reverse depth-first order, each node's coordinates reversed, so every
    subtree is one contiguous range and ``P H P^T = L L^T`` has no fill. The
    fields are those of the solver's sparse mass-matrix plan.
    """

    permutation: np.ndarray
    inverse_permutation: np.ndarray
    row_offsets: np.ndarray
    columns: np.ndarray
    entry_rows: np.ndarray
    lookup: np.ndarray
    """Packed entry of factor (row, column), ``-1`` where structurally zero (lower triangle only)."""
    physical_lookup: np.ndarray
    """Packed entry of physical (row, column), symmetric."""
    dof_joint: np.ndarray
    ancestor_mask: np.ndarray
    """Factor-coordinate support of each physical coordinate's node."""
    joint_ancestor_mask: np.ndarray
    """Factor-coordinate support of each node (its own and its ancestors' coordinates)."""
    column_offsets: np.ndarray
    column_entries: np.ndarray
    levels: np.ndarray
    """Dependency level of each factor column."""
    factor_level_offsets: np.ndarray
    factor_level_columns: np.ndarray
    factor_level_entry_offsets: np.ndarray
    factor_level_entries: np.ndarray

    @property
    def dof_count(self) -> int:
        return int(self.permutation.size)

    @property
    def nonzero_count(self) -> int:
        return int(self.columns.size)


@dataclass(frozen=True, eq=False)
class SupportClasses:
    """Row support patterns over a tree's factor coordinates: every endpoint and endpoint pair, and every limit.

    ``body_class[node]`` is the node's pattern index among the endpoint classes
    (index 0, the empty pattern, is the world); ``pair_class[a, b]`` the union's
    index among all patterns; ``class_dof[k]`` the factor coordinates of pattern
    ``k`` (``-1`` padded); ``limit_class[dof]`` a physical coordinate's pattern.
    """

    body_class: np.ndarray
    pair_class: np.ndarray
    class_dof: np.ndarray
    limit_class: np.ndarray


# ---------------------------------------------------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class TreePlan:
    """Every traversal structure of one joint forest, built once on the host (see the module docstring)."""

    parent: np.ndarray
    """Parent of each node, ``-1`` at a root."""
    dofs: np.ndarray
    """Coordinates of each node."""
    dof_start: np.ndarray
    """First coordinate of each node (and the total, last)."""
    order: np.ndarray
    """A parent-first order of the nodes: ascending index when the input is parent-first."""
    roots: np.ndarray
    """Roots, ascending."""
    root_of: np.ndarray
    """The root of each node's tree."""
    depth: np.ndarray
    """Edges from each node to its root."""
    children_offsets: np.ndarray
    children: np.ndarray
    """Children of each node, descending index (the serial leaf-to-root accumulation order)."""
    level_offsets: np.ndarray
    level_nodes: np.ndarray
    """Nodes of each depth, ascending index."""
    jump: np.ndarray
    """``jump[r, i]``: the ``2**r``-th ancestor of node ``i``, ``-1`` past the root; ``rounds`` rows."""
    subtree_size: np.ndarray
    dfs_order: np.ndarray
    """Node at each depth-first position (children and roots ascending)."""
    dfs_position: np.ndarray
    """Depth-first position of each node."""
    dfs_end: np.ndarray
    """Per position, one past the last position of its subtree."""
    heavy: np.ndarray
    """Each node's child with the largest subtree (the lowest index on a tie), ``-1`` at a leaf."""
    chain_of: np.ndarray
    """Heavy chain of each node; chains are numbered by their heads, ascending."""
    chain_position: np.ndarray
    """Position of each node along its chain, from the head."""
    chain_heads: np.ndarray
    chain_lengths: np.ndarray
    chain_levels: np.ndarray
    """Light level of each chain: the light edges above its head."""
    light_offsets: np.ndarray
    light_children: np.ndarray
    """Children of each node but its heavy child, descending index."""

    # --- construction -------------------------------------------------------------------------------------------
    @classmethod
    def build(cls, parent, dofs=None) -> TreePlan:
        """Build the plan of the forest given by each node's ``parent`` and coordinate count ``dofs``.

        Args:
            parent: Parent node of each node, ``-1`` at a root; any acyclic forest.
            dofs: Coordinates of each node (default one each).

        Raises:
            ValueError: The parents name a node out of range or form a cycle.
        """
        parent = np.asarray(parent, dtype=np.int64).ravel()
        count = int(parent.size)
        dofs = np.ones(count, dtype=np.int64) if dofs is None else np.asarray(dofs, dtype=np.int64).ravel()
        if dofs.shape != parent.shape or np.any(dofs < 0):
            raise ValueError("A tree plan needs a nonnegative coordinate count per node")
        if np.any(parent < -1) or np.any(parent >= count) or np.any(parent == np.arange(count)):
            raise ValueError("A tree plan's parents must name other nodes or -1")
        nodes = np.arange(count, dtype=np.int64)
        dof_start = np.zeros(count + 1, dtype=np.int64)
        np.cumsum(dofs, out=dof_start[1:])

        # Depth and the jump table by pointer doubling: after r rounds every node knows its 2**r-th ancestor.
        depth = (parent >= 0).astype(np.int64)
        ancestor = parent.copy()
        table = [parent.copy()]
        for _ in range(max(1, count.bit_length()) + 1):
            live = ancestor >= 0
            if not np.any(live):
                break
            depth = depth + np.where(live, depth[np.maximum(ancestor, 0)], 0)
            ancestor = np.where(live, ancestor[np.maximum(ancestor, 0)], -1)
            table.append(ancestor.copy())
        else:
            raise ValueError("A tree plan's parents form a cycle")
        if np.any(ancestor >= 0):
            raise ValueError("A tree plan's parents form a cycle")
        max_depth = int(depth.max()) if count else 0
        rounds = max(1, max_depth.bit_length())
        jump = np.full((rounds, count), -1, dtype=np.int64)
        for r in range(min(rounds, len(table))):
            jump[r] = table[r]

        roots = np.flatnonzero(parent < 0)
        level_offsets, level_nodes = _group_by(depth, max_depth + 1)
        # A parent-first order: the input order when every parent precedes its children, else by depth.
        parent_first = bool(np.all(parent < nodes))
        order = nodes.copy() if parent_first else level_nodes.copy()

        children_mask = parent >= 0
        child_nodes = nodes[children_mask]
        by_parent = np.lexsort((-child_nodes, parent[children_mask]))
        children = child_nodes[by_parent]
        children_offsets = np.zeros(count + 1, dtype=np.int64)
        np.cumsum(np.bincount(parent[children_mask], minlength=count), out=children_offsets[1:])

        # Subtree sizes, deepest level first; each node's root, shallowest first.
        size = np.ones(count, dtype=np.int64)
        for level in range(max_depth, 0, -1):
            level_set = level_nodes[level_offsets[level] : level_offsets[level + 1]]
            np.add.at(size, parent[level_set], size[level_set])
        root_of = nodes.copy()
        for level in range(1, max_depth + 1):
            level_set = level_nodes[level_offsets[level] : level_offsets[level + 1]]
            root_of[level_set] = root_of[parent[level_set]]

        # Depth-first positions: a child follows its parent and the subtrees of its lower siblings.
        position = np.zeros(count, dtype=np.int64)
        position[roots] = np.cumsum(size[roots]) - size[roots]
        for level in range(1, max_depth + 1):
            level_set = level_nodes[level_offsets[level] : level_offsets[level + 1]]
            # level_set ascends, so a parent's children ascend within it.
            by_family = np.argsort(parent[level_set], kind="stable")
            family = level_set[by_family]
            sizes = size[family]
            inclusive = np.cumsum(sizes)
            first = np.r_[True, parent[family][1:] != parent[family][:-1]]
            group_start = np.maximum.accumulate(np.where(first, np.arange(family.size), 0))
            before = inclusive - sizes - (inclusive[group_start] - sizes[group_start])
            position[family] = position[parent[family]] + 1 + before
        dfs_order = np.empty(count, dtype=np.int64)
        dfs_order[position] = nodes
        dfs_end = position[dfs_order] + size[dfs_order]

        # Heavy children: the largest subtree, the lowest index on a tie.
        heavy = np.full(count, -1, dtype=np.int64)
        if child_nodes.size:
            pick = np.lexsort((child_nodes, -size[child_nodes], parent[children_mask]))
            ordered = child_nodes[pick]
            ordered_parent = parent[ordered]
            leading = np.r_[True, ordered_parent[1:] != ordered_parent[:-1]]
            heavy[ordered_parent[leading]] = ordered[leading]
        is_heavy = np.zeros(count, dtype=bool)
        is_heavy[heavy[heavy >= 0]] = True
        heads = nodes[~is_heavy]
        chain_of = np.full(count, -1, dtype=np.int64)
        chain_of[heads] = np.arange(heads.size)
        chain_position = np.zeros(count, dtype=np.int64)
        chain_level = np.zeros(heads.size, dtype=np.int64)
        for level in range(1, max_depth + 1):
            level_set = level_nodes[level_offsets[level] : level_offsets[level + 1]]
            follows = is_heavy[level_set]
            continued = level_set[follows]
            chain_of[continued] = chain_of[parent[continued]]
            chain_position[continued] = chain_position[parent[continued]] + 1
            started = level_set[~follows]
            chain_level[chain_of[started]] = chain_level[chain_of[parent[started]]] + 1
        chain_lengths = np.bincount(chain_of, minlength=heads.size).astype(np.int64)

        light_mask = children_mask & ~is_heavy
        light_nodes = nodes[light_mask]
        light = light_nodes[np.lexsort((-light_nodes, parent[light_mask]))]
        light_offsets = np.zeros(count + 1, dtype=np.int64)
        np.cumsum(np.bincount(parent[light_mask], minlength=count), out=light_offsets[1:])

        values = {
            "parent": parent,
            "dofs": dofs,
            "dof_start": dof_start,
            "order": order,
            "roots": roots,
            "root_of": root_of,
            "depth": depth,
            "children_offsets": children_offsets,
            "children": children,
            "level_offsets": level_offsets,
            "level_nodes": level_nodes,
            "jump": jump,
            "subtree_size": size,
            "dfs_order": dfs_order,
            "dfs_position": position,
            "dfs_end": dfs_end,
            "heavy": heavy,
            "chain_of": chain_of,
            "chain_position": chain_position,
            "chain_heads": heads,
            "chain_lengths": chain_lengths,
            "chain_levels": chain_level,
            "light_offsets": light_offsets,
            "light_children": light,
        }
        for name, value in values.items():
            values[name] = value.astype(np.int32)
            values[name].setflags(write=False)
        return cls(**values)

    # --- sizes ---------------------------------------------------------------------------------------------------
    @property
    def count(self) -> int:
        """Nodes."""
        return int(self.parent.size)

    @property
    def dof_count(self) -> int:
        """Coordinates."""
        return int(self.dof_start[-1])

    @property
    def max_depth(self) -> int:
        """Edges on the longest root-to-leaf path."""
        return int(self.depth.max()) if self.count else 0

    @property
    def rounds(self) -> int:
        """Pointer-jumping rounds that cover every root path."""
        return int(self.jump.shape[0])

    @property
    def level_widths(self) -> np.ndarray:
        """Nodes of each depth."""
        return np.diff(self.level_offsets)

    @property
    def dof_node(self) -> np.ndarray:
        """The node owning each coordinate."""
        return np.repeat(np.arange(self.count, dtype=np.int32), self.dofs)

    def parent_slots(self, slot: np.ndarray) -> np.ndarray:
        """Each node's parent's slot, ``slot[parent]``, or ``-1`` at a root.

        A forest of articulations scans body-indexed values through its joints: ``slot`` is each joint's
        child body, so a joint's parent slot is its parent joint's child body.
        """
        slot = np.asarray(slot, dtype=np.int32)
        return np.where(self.parent >= 0, slot[np.maximum(self.parent, 0)], -1).astype(np.int32)

    @property
    def parent_first(self) -> bool:
        """Whether every parent precedes its children (the order the serial loops walk)."""
        return bool(np.all(self.parent < np.arange(self.count)))

    def tree_depths(self) -> np.ndarray:
        """Nodes on the longest root-to-leaf path of each tree, per root (a tree's depth in joints)."""
        deepest = np.zeros(self.count, dtype=np.int64)
        np.maximum.at(deepest, self.root_of, self.depth.astype(np.int64) + 1)
        return deepest[self.roots]

    # --- masks ---------------------------------------------------------------------------------------------------
    @functools.cached_property
    def ancestor_mask(self) -> np.ndarray:
        """Bit ``a`` of node ``i``'s mask is set when ``a`` is ``i`` or one of its ancestors (64 nodes at most)."""
        if self.count > 64:
            raise ValueError("Node ancestor masks cover at most 64 nodes")
        mask = np.zeros(self.count, dtype=np.uint64)
        bits = np.left_shift(np.uint64(1), np.arange(self.count, dtype=np.uint64))
        for level in range(self.max_depth + 1):
            level_set = self.level_nodes[self.level_offsets[level] : self.level_offsets[level + 1]]
            above = np.where(self.parent[level_set] >= 0, mask[np.maximum(self.parent[level_set], 0)], np.uint64(0))
            mask[level_set] = above | bits[level_set]
        mask.setflags(write=False)
        return mask

    def ancestor_matrix(self) -> np.ndarray:
        """``matrix[i, a]`` tells whether ``a`` is ``i`` or one of its ancestors (a dense square of the nodes)."""
        matrix = np.zeros((self.count, self.count), dtype=bool)
        for level in range(self.max_depth + 1):
            level_set = self.level_nodes[self.level_offsets[level] : self.level_offsets[level + 1]]
            above = self.parent[level_set]
            matrix[level_set[above >= 0]] = matrix[above[above >= 0]]
            matrix[level_set, level_set] = True
        return matrix

    @functools.cached_property
    def coordinate_mask(self) -> np.ndarray:
        """Bits of the coordinates of each node and its ancestors, from its root's first coordinate (64 at most)."""
        first = self.dof_start[self.root_of].astype(np.int64)
        low = self.dof_start[:-1].astype(np.int64) - first
        high = low + self.dofs
        if np.any(high > 64):
            raise ValueError("Coordinate masks cover trees of at most 64 coordinates")
        one = np.uint64(1)
        own = (np.left_shift(one, high.astype(np.uint64), dtype=np.uint64) - one) ^ (
            np.left_shift(one, low.astype(np.uint64), dtype=np.uint64) - one
        )
        own = np.where(high >= 64, ~(np.left_shift(one, low.astype(np.uint64)) - one), own).astype(np.uint64)
        mask = np.zeros(self.count, dtype=np.uint64)
        for level in range(self.max_depth + 1):
            level_set = self.level_nodes[self.level_offsets[level] : self.level_offsets[level + 1]]
            above = np.where(self.parent[level_set] >= 0, mask[np.maximum(self.parent[level_set], 0)], np.uint64(0))
            mask[level_set] = above | own[level_set]
        mask.setflags(write=False)
        return mask

    # --- heavy chains --------------------------------------------------------------------------------------------
    def chain_packing(self, warps: int = 2) -> ChainPacking | None:
        """Pack the padded heavy chains first-fit into ``warps`` warps, widest first; ``None`` when they do not fit.

        A chain pads to the next power of two; no padded chain may cross a warp.
        """
        widths = np.left_shift(1, np.ceil(np.log2(np.maximum(self.chain_lengths, 1))).astype(np.int64))
        if self.count == 0 or np.any(widths > 32):
            return None
        meta = np.full((32 * warps, 5), -1, dtype=np.int32)
        meta[:, 1] = np.arange(32 * warps)
        meta[:, 2] = 0
        used = [0] * warps
        members = _group_by(self.chain_of, self.chain_heads.size)
        for chain in sorted(range(self.chain_heads.size), key=lambda c: (-int(widths[c]), int(self.chain_heads[c]))):
            candidates = [warp for warp in range(warps) if used[warp] + widths[chain] <= 32]
            if not candidates:
                return None
            start = 32 * candidates[0] + used[candidates[0]]
            used[candidates[0]] += int(widths[chain])
            chain_nodes = members[1][members[0][chain] : members[0][chain + 1]]
            chain_nodes = chain_nodes[np.argsort(self.chain_position[chain_nodes])]
            head = int(self.chain_heads[chain])
            for offset, node in enumerate(chain_nodes):
                meta[start + offset] = (
                    node,
                    start,
                    self.chain_lengths[chain],
                    self.chain_levels[chain],
                    self.parent[head],
                )
        levels = int(self.chain_levels.max()) + 1
        stage_width = np.zeros(levels, dtype=np.int32)
        np.maximum.at(stage_width, self.chain_levels, widths.astype(np.int32))
        meta.setflags(write=False)
        stage_width.setflags(write=False)
        return ChainPacking(meta, levels, stage_width, int(widths.max()), warps)

    # --- the joint-space inertia's factor -------------------------------------------------------------------------
    @functools.cached_property
    def factor(self) -> FactorPattern:
        """The fill-free factor pattern of the joint-space inertia (64 coordinates at most)."""
        n = self.dof_count
        if not 1 <= n <= 64 or np.any(self.dofs > 64):
            raise ValueError("Factor patterns cover 1 to 64 coordinates")
        start = self.dof_start.astype(np.int64)
        # Leaf-first: the reverse depth-first order of the nodes, each node's coordinates reversed.
        permutation = np.concatenate(
            [np.arange(start[node + 1] - 1, start[node] - 1, -1) for node in self.dfs_order[::-1]]
        ).astype(np.int32)
        inverse = np.argsort(permutation).astype(np.int32)
        dof_joint = np.repeat(np.arange(self.count, dtype=np.int32), self.dofs)
        # Support of each node in factor coordinates: the coordinates of its ancestors and itself.
        support = np.zeros((self.count, n), dtype=bool)
        support[:, inverse] = self.ancestor_matrix()[:, dof_joint]
        weights = np.left_shift(np.uint64(1), np.arange(n, dtype=np.uint64))
        joint_masks = np.bitwise_or.reduce(np.where(support, weights, np.uint64(0)), axis=1).astype(np.uint64)
        present = np.tril(support[dof_joint[permutation]].T)  # [row, col]: col's node is supported by row
        rows, cols = np.nonzero(present)
        lookup = np.full((n, n), -1, dtype=np.int32)
        lookup[rows, cols] = np.arange(rows.size, dtype=np.int32)
        row_offsets = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(np.bincount(rows, minlength=n), out=row_offsets[1:])
        column_order = np.lexsort((rows, cols))
        column_offsets = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(np.bincount(cols, minlength=n), out=column_offsets[1:])
        physical_lookup = np.full((n, n), -1, dtype=np.int32)
        a, b = permutation[rows], permutation[cols]
        physical_lookup[a, b] = physical_lookup[b, a] = np.arange(rows.size, dtype=np.int32)
        # A pivot depends on the earlier columns its row holds; sibling subtrees share a level.
        levels = np.zeros(n, dtype=np.int32)
        for col in range(n):
            previous = np.flatnonzero(lookup[col, :col] >= 0)
            levels[col] = 0 if not previous.size else 1 + int(levels[previous].max())
        level_count = int(levels.max()) + 1
        level_offsets = np.zeros(level_count + 1, dtype=np.int32)
        np.cumsum(np.bincount(levels, minlength=level_count), out=level_offsets[1:])
        off = np.flatnonzero(rows != cols)
        entries = off[np.lexsort((rows[off], cols[off], levels[cols[off]]))]
        entry_offsets = np.zeros(level_count + 1, dtype=np.int32)
        np.cumsum(np.bincount(levels[cols[entries]], minlength=level_count), out=entry_offsets[1:])
        values = {
            "permutation": permutation,
            "inverse_permutation": inverse,
            "row_offsets": row_offsets,
            "columns": cols.astype(np.int32),
            "entry_rows": rows.astype(np.int32),
            "lookup": lookup,
            "physical_lookup": physical_lookup,
            "dof_joint": dof_joint,
            "ancestor_mask": joint_masks[dof_joint],
            "joint_ancestor_mask": joint_masks,
            "column_offsets": column_offsets,
            "column_entries": column_order.astype(np.int32),
            "levels": levels,
            "factor_level_offsets": level_offsets,
            "factor_level_columns": np.argsort(levels, kind="stable").astype(np.int32),
            "factor_level_entry_offsets": entry_offsets,
            "factor_level_entries": entries.astype(np.int32),
        }
        for value in values.values():
            value.setflags(write=False)
        return FactorPattern(**values)

    def support_classes(self, body_masks, max_row_dofs: int) -> SupportClasses | None:
        """Classify the row supports of a tree whose bodies have the factor-coordinate ``body_masks``.

        Every body is a possible contact endpoint, so the classes are the bodies'
        masks with the world's empty one, their pairwise unions and every
        coordinate's limit support. ``None`` when a support exceeds ``max_row_dofs``.
        """
        factor = self.factor
        body_masks = np.asarray(body_masks, dtype=np.uint64)
        if body_masks.ndim != 1 or not 1 <= max_row_dofs <= factor.dof_count:
            raise ValueError("Row supports need one-dimensional masks and a valid row width")
        if factor.dof_count < 64 and np.any(body_masks >> np.uint64(factor.dof_count)):
            raise ValueError("A row support exceeds the tree's coordinates")
        classes = np.unique(np.concatenate((np.zeros(1, dtype=np.uint64), body_masks)))
        pairs = classes[:, None] | classes[None, :]
        masks = np.unique(np.concatenate((pairs.ravel(), factor.ancestor_mask)))
        bits = ((masks[:, None] >> np.arange(factor.dof_count, dtype=np.uint64)) & np.uint64(1)).astype(bool)
        if np.any(bits.sum(axis=1) > max_row_dofs):
            return None
        class_dof = np.full((masks.size, max_row_dofs), -1, dtype=np.int32)
        rank = np.cumsum(bits, axis=1) - 1
        which, dof = np.nonzero(bits)
        class_dof[which, rank[which, dof]] = dof
        return SupportClasses(
            body_class=np.searchsorted(classes, body_masks).astype(np.int32),
            pair_class=np.searchsorted(masks, pairs).astype(np.int32),
            class_dof=class_dof,
            limit_class=np.searchsorted(masks, factor.ancestor_mask).astype(np.int32),
        )
