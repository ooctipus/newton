# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Color free-body contact pairs for conflict-free, multi-block PGS sweeps."""

import warp as wp


@wp.func_native("union { int bits; float value; } result; result.bits = bits; return result.value;")
def _float_from_bits(bits: int) -> float: ...


@wp.func
def _endpoint_a(packed: int):
    return packed >> 16


@wp.func
def _endpoint_b(packed: int):
    value = packed & 65535
    if value >= 32768:
        value -= 65536
    return value


@wp.func
def _parent(packed: int):
    value = int(wp.uint32(packed) >> wp.uint32(16))
    if value == 65535:
        value = -1
    return value


@wp.kernel(enable_backward=False)
def _make_keys(
    max_rows: int,
    counts: wp.array[int],
    contact_end: wp.array[int],
    meta: wp.array2d[int],
    keys: wp.array[wp.int64],
    indices: wp.array[int],
    lengths: wp.array2d[int],
    row_pair: wp.array2d[int],
):
    flat = wp.tid()
    world = flat // max_rows
    row = flat % max_rows
    end = wp.min(wp.min(counts[world], contact_end[world]), max_rows)
    pair = wp.int64(4294967295)
    length = int(0)
    if row < end and (meta[world, row * 4 + 3] & 65535) == 0:
        endpoints = meta[world, row * 4]
        a = _endpoint_a(endpoints)
        b = _endpoint_b(endpoints)
        lo = wp.min(a, b)
        hi = wp.max(a, b)
        pair = (wp.int64(lo + 1) << wp.int64(16)) | wp.int64(hi + 1)
        length = 1
        if row + 2 < end:
            first = meta[world, (row + 1) * 4 + 3]
            second = meta[world, (row + 2) * 4 + 3]
            if (first & 65535) == 2 and (second & 65535) == 2 and _parent(first) == row and _parent(second) == row:
                length = 3
    keys[flat] = (wp.int64(world) << wp.int64(32)) | pair
    indices[flat] = row
    lengths[world, row] = length
    row_pair[world, row] = -1


@wp.kernel(enable_backward=False)
def _group_and_color(
    max_rows: int,
    max_colors: int,
    sorted_keys: wp.array[wp.int64],
    sorted_indices: wp.array[int],
    lengths: wp.array2d[int],
    body_masks: wp.array2d[wp.uint32],
    row_indices: wp.array2d[int],
    row_pair: wp.array2d[int],
    pair_offsets: wp.array2d[int],
    pair_a: wp.array2d[int],
    pair_b: wp.array2d[int],
    pair_color: wp.array2d[int],
    unit_count: wp.array[int],
    pair_count: wp.array[int],
    color_count: wp.array[int],
    color_counts: wp.array2d[int],
    tail_count: wp.array[int],
    color_offsets: wp.array2d[int],
    color_pairs: wp.array2d[int],
    cursors: wp.array2d[int],
):
    world = wp.tid()
    base = world * max_rows
    pairs = int(0)
    units = int(0)
    previous = wp.int64(-1)
    for index in range(max_rows):
        key = sorted_keys[base + index]
        local_key = key & wp.int64(4294967295)
        if local_key == wp.int64(4294967295):
            break
        if key != previous:
            pair_offsets[world, pairs] = units
            pair_a[world, pairs] = int(local_key >> wp.int64(16)) - 1
            pair_b[world, pairs] = int(local_key & wp.int64(65535)) - 1
            pairs += 1
            previous = key
        row = sorted_indices[base + index]
        row_indices[world, units] = row
        for child in range(lengths[world, row]):
            row_pair[world, row + child] = pairs - 1
        units += 1
    pair_offsets[world, pairs] = units
    unit_count[world] = units
    pair_count[world] = pairs

    for color in range(max_colors):
        color_counts[world, color] = 0
    tail = int(0)
    for pair_index in range(pairs):
        a = pair_a[world, pair_index]
        b = pair_b[world, pair_index]
        used = wp.uint32(0)
        if a >= 0:
            used = body_masks[world, a]
        if b >= 0:
            used = used | body_masks[world, b]
        chosen = max_colors
        for color in range(max_colors):
            bit = wp.uint32(1) << wp.uint32(color)
            if (used & bit) == wp.uint32(0):
                chosen = color
                break
        pair_color[world, pair_index] = chosen
        if chosen < max_colors:
            bit = wp.uint32(1) << wp.uint32(chosen)
            if a >= 0:
                body_masks[world, a] = body_masks[world, a] | bit
            if b >= 0:
                body_masks[world, b] = body_masks[world, b] | bit
            color_counts[world, chosen] += 1
        else:
            tail += 1

    offset = int(0)
    active_colors = int(0)
    for color in range(max_colors):
        color_offsets[world, color] = offset
        cursors[world, color] = offset
        count = color_counts[world, color]
        offset += count
        if count > 0:
            active_colors += 1
    color_offsets[world, max_colors] = offset
    cursors[world, max_colors] = offset
    color_offsets[world, max_colors + 1] = offset + tail
    color_count[world] = active_colors
    tail_count[world] = tail
    for pair_index in range(pairs):
        color = pair_color[world, pair_index]
        index = cursors[world, color]
        color_pairs[world, index] = pair_index
        cursors[world, color] = index + 1


@wp.struct
class _SolveData:
    world_dof_indices: wp.array2d[int]
    row_indices: wp.array2d[int]
    unit_lengths: wp.array2d[int]
    pair_offsets: wp.array2d[int]
    pair_a: wp.array2d[int]
    pair_b: wp.array2d[int]
    pair_count: wp.array[int]
    static_pair_count: wp.array[int]
    color_offsets: wp.array2d[int]
    color_pairs: wp.array2d[int]
    meta: wp.array2d[int]
    impulses: wp.array2d[float]
    J_a: wp.array3d[float]
    J_b: wp.array3d[float]
    MiJt_a: wp.array3d[float]
    MiJt_b: wp.array3d[float]
    mu: wp.array2d[float]
    row_w: wp.array2d[float]
    velocity: wp.array[float]
    body_degree: wp.array2d[int]
    velocity_delta: wp.array2d[float]
    omega: float
    regularize: int
    friction_active: int
    direct_static_pairs: int
    cache_coefficients: int
    coefficients: wp.array2d[wp.vec2]
    coefficient_ready: wp.array[int]
    coefficient_cache_misses: wp.array[int]


@wp.func
def _apply_local_response(
    data: _SolveData,
    world: int,
    row: int,
    delta: float,
    pair_a: int,
    valid_a: int,
    valid_b: int,
    velocity_a: wp.spatial_vector,
    velocity_b: wp.spatial_vector,
):
    endpoints = data.meta[world, row * 4]
    a = _endpoint_a(endpoints)
    b = _endpoint_b(endpoints)
    for component in range(6):
        if a >= 0:
            if a == pair_a:
                if (valid_a & (1 << component)) != 0:
                    velocity_a[component] += data.MiJt_a[world, row, component] * delta
            else:
                if (valid_b & (1 << component)) != 0:
                    velocity_b[component] += data.MiJt_a[world, row, component] * delta
        if b >= 0:
            if b == pair_a:
                if (valid_a & (1 << component)) != 0:
                    velocity_a[component] += data.MiJt_b[world, row, component] * delta
            else:
                if (valid_b & (1 << component)) != 0:
                    velocity_b[component] += data.MiJt_b[world, row, component] * delta
    return velocity_a, velocity_b


@wp.func
def _solve_pair(data: _SolveData, world: int, pair: int):
    pair_a = data.pair_a[world, pair]
    pair_b = data.pair_b[world, pair]
    velocity_a = wp.spatial_vector()
    velocity_b = wp.spatial_vector()
    valid_a = int(0)
    valid_b = int(0)
    # A color owns both endpoints until this pair completes. Keep all row and
    # sibling corrections local, including the same-endpoint case.
    for component in range(6):
        if pair_a >= 0:
            dof = data.world_dof_indices[world, pair_a + component]
            if dof >= 0:
                velocity_a[component] = data.velocity[dof]
                valid_a = valid_a | (1 << component)
        if pair_b >= 0 and pair_b != pair_a:
            dof = data.world_dof_indices[world, pair_b + component]
            if dof >= 0:
                velocity_b[component] = data.velocity[dof]
                valid_b = valid_b | (1 << component)
    for unit in range(data.pair_offsets[world, pair], data.pair_offsets[world, pair + 1]):
        start = data.row_indices[world, unit]
        for child in range(data.unit_lengths[world, start]):
            row = start + child
            row_type = data.meta[world, row * 4 + 3] & 65535
            if row_type == 2 and data.friction_active == 0:
                data.impulses[world, row] = 0.0
                continue
            effective_inverse = _float_from_bits(data.meta[world, row * 4 + 1])
            if effective_inverse <= 0.0:
                continue
            endpoints = data.meta[world, row * 4]
            a = _endpoint_a(endpoints)
            b = _endpoint_b(endpoints)
            jv = float(0.0)
            for component in range(6):
                if a >= 0:
                    velocity = velocity_a[component]
                    if a != pair_a:
                        velocity = velocity_b[component]
                    jv += data.J_a[world, row, component] * velocity
                if b >= 0:
                    velocity = velocity_a[component]
                    if b != pair_a:
                        velocity = velocity_b[component]
                    jv += data.J_b[world, row, component] * velocity
            residual = jv + _float_from_bits(data.meta[world, row * 4 + 2])
            old = data.impulses[world, row]
            delta = -residual * effective_inverse
            if row_type == 0:
                weight = float(1.0)
                if data.regularize != 0:
                    weight = data.row_w[world, row]
                delta = -residual * effective_inverse * weight - (1.0 - weight) * old
            impulse = old + data.omega * delta
            if row_type == 0:
                impulse = wp.max(impulse, 0.0)
            else:
                parent = _parent(data.meta[world, row * 4 + 3])
                radius = wp.max(data.mu[world, row] * data.impulses[world, parent], 0.0)
                if radius <= 0.0:
                    impulse = 0.0
                else:
                    sibling = parent + 1
                    if row == parent + 1:
                        sibling = parent + 2
                    data.impulses[world, row] = impulse
                    other = data.impulses[world, sibling]
                    magnitude = wp.sqrt(impulse * impulse + other * other)
                    if magnitude > radius:
                        scale = radius / magnitude
                        impulse *= scale
                        sibling_new = other * scale
                        data.impulses[world, sibling] = sibling_new
                        velocity_a, velocity_b = _apply_local_response(
                            data,
                            world,
                            sibling,
                            sibling_new - other,
                            pair_a,
                            valid_a,
                            valid_b,
                            velocity_a,
                            velocity_b,
                        )
            data.impulses[world, row] = impulse
            change = impulse - old
            if change != 0.0:
                velocity_a, velocity_b = _apply_local_response(
                    data, world, row, change, pair_a, valid_a, valid_b, velocity_a, velocity_b
                )
    for component in range(6):
        if (valid_a & (1 << component)) != 0:
            dof = data.world_dof_indices[world, pair_a + component]
            data.velocity[dof] = velocity_a[component]
        if (valid_b & (1 << component)) != 0:
            dof = data.world_dof_indices[world, pair_b + component]
            data.velocity[dof] = velocity_b[component]


@wp.func_native(r"""
#if defined(__CUDA_ARCH__)
    const unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int pair_a = *wp::address(data.pair_a, world, pair);
    const int pair_b = *wp::address(data.pair_b, world, pair);
    const int component = lane < 6 ? lane : lane - 6;
    const int owned = lane < 6 ? pair_a : pair_b;
    int global_dof = -1;
    if (lane < 12 && owned >= 0 && (lane < 6 || pair_b != pair_a))
        global_dof = *wp::address(data.world_dof_indices, world, owned + component);
    float velocity = global_dof >= 0 ? *wp::address(data.velocity, global_dof) : 0.0f;
    float physical_delta = 0.0f;
    const float degree = mass_split && global_dof >= 0
        ? (float)*wp::address(data.body_degree, world, owned) : 1.0f;
    // Readiness is device-side too: replaying a topology-only graph must not
    // consume coefficients from its previous topology or regularization mode.
    const bool cached = data.cache_coefficients
        && *wp::address(data.coefficient_ready, world) == 1 + (data.regularize != 0);
    if (data.cache_coefficients && !cached && lane == 0)
        atomicExch(wp::address(data.coefficient_cache_misses, world), 1);
    const int begin = *wp::address(data.pair_offsets, world, pair);
    const int end = *wp::address(data.pair_offsets, world, pair + 1);
    for (int unit = begin; unit < end; ++unit) {
        const int start = *wp::address(data.row_indices, world, unit);
        const int length = *wp::address(data.unit_lengths, world, start);
        for (int child = 0; child < length; ++child) {
            const int row = start + child;
            const int packed_type = *wp::address(data.meta, world, 4 * row + 3);
            const int type = packed_type & 65535;
            // Only lane zero ever reads or writes impulses, including siblings.
            if (type == 2 && !data.friction_active) {
                if (lane == 0) *wp::address(data.impulses, world, row) = 0.0f;
                continue;
            }
            const float effective_inverse = __int_as_float(*wp::address(data.meta, world, 4 * row + 1));
            if (effective_inverse <= 0.0f) continue;
            const int endpoints = *wp::address(data.meta, world, 4 * row);
            const int a = endpoints >> 16;
            const int b = (int)(short)(endpoints & 65535);
            float jv = 0.0f;
            float extra_diagonal = 0.0f;
            if (global_dof >= 0) {
                if (a == owned) jv += *wp::address(data.J_a, world, row, component) * velocity;
                if (b == owned) jv += *wp::address(data.J_b, world, row, component) * velocity;
                if (mass_split && !cached) {
                    // Combining sides before multiplication includes self-pair cross terms.
                    float jacobian = 0.0f, response = 0.0f;
                    if (a == owned) {
                        jacobian += *wp::address(data.J_a, world, row, component);
                        response += *wp::address(data.MiJt_a, world, row, component);
                    }
                    if (b == owned) {
                        jacobian += *wp::address(data.J_b, world, row, component);
                        response += *wp::address(data.MiJt_b, world, row, component);
                    }
                    extra_diagonal = (degree - 1.0f) * jacobian * response;
                }
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                jv += __shfl_down_sync(mask, jv, offset);
                if (mass_split && !cached) extra_diagonal += __shfl_down_sync(mask, extra_diagonal, offset);
            }
            const int parent_bits = (int)((unsigned int)packed_type >> 16);
            const int parent = parent_bits == 65535 ? -1 : parent_bits;
            const int sibling = row == parent + 1 ? parent + 2 : parent + 1;
            float change = 0.0f;
            float sibling_change = 0.0f;
            if (lane == 0) {
                const float residual = jv + __int_as_float(*wp::address(data.meta, world, 4 * row + 2));
                const float old = *wp::address(data.impulses, world, row);
                float delta;
                if (cached) {
                    const wp::vec2 coefficients = *wp::address(data.coefficients, world, row);
                    delta = -coefficients[0] * residual - coefficients[1] * old;
                } else {
                    delta = -residual * effective_inverse;
                    if (type == 0) {
                        const float weight = data.regularize ? *wp::address(data.row_w, world, row) : 1.0f;
                        delta = -residual * effective_inverse * weight - (1.0f - weight) * old;
                    }
                    if (mass_split) {
                        const float old_diagonal = 1.0f / effective_inverse;
                        const float split_diagonal = old_diagonal + extra_diagonal;
                        delta = -residual / split_diagonal;
                        if (type == 0 && data.regularize) {
                            const float weight = *wp::address(data.row_w, world, row);
                            // Equivalent to g=Dold*(1-w)/w without overflow as w tends to zero.
                            const float weighted_g = old_diagonal * (1.0f - weight);
                            delta = weight <= 0.0f ? -old
                                : -(weight * residual + weighted_g * old)
                                    / (weight * split_diagonal + weighted_g);
                        }
                    }
                }
                float impulse = old + data.omega * delta;
                if (type == 0) {
                    impulse = fmaxf(impulse, 0.0f);
                } else {
                    const float radius = fmaxf(*wp::address(data.mu, world, row)
                        * *wp::address(data.impulses, world, parent), 0.0f);
                    if (radius <= 0.0f) {
                        impulse = 0.0f;
                    } else {
                        const float other = *wp::address(data.impulses, world, sibling);
                        const float magnitude = sqrtf(impulse * impulse + other * other);
                        if (magnitude > radius) {
                            const float scale = radius / magnitude;
                            impulse *= scale;
                            const float sibling_new = other * scale;
                            *wp::address(data.impulses, world, sibling) = sibling_new;
                            sibling_change = sibling_new - other;
                        }
                    }
                }
                *wp::address(data.impulses, world, row) = impulse;
                change = impulse - old;
            }
            change = __shfl_sync(mask, change, 0);
            sibling_change = __shfl_sync(mask, sibling_change, 0);
            // Preserve the scalar law's sibling response before current response.
            if (sibling_change != 0.0f) {
                const int other_endpoints = *wp::address(data.meta, world, 4 * sibling);
                const int other_a = other_endpoints >> 16;
                const int other_b = (int)(short)(other_endpoints & 65535);
                if (global_dof >= 0) {
                    if (other_a == owned) {
                        const float response = *wp::address(data.MiJt_a, world, sibling, component) * sibling_change;
                        velocity += degree * response;
                        if (mass_split) physical_delta += response;
                    }
                    if (other_b == owned) {
                        const float response = *wp::address(data.MiJt_b, world, sibling, component) * sibling_change;
                        velocity += degree * response;
                        if (mass_split) physical_delta += response;
                    }
                }
            }
            if (change != 0.0f && global_dof >= 0) {
                if (a == owned) {
                    const float response = *wp::address(data.MiJt_a, world, row, component) * change;
                    velocity += degree * response;
                    if (mass_split) physical_delta += response;
                }
                if (b == owned) {
                    const float response = *wp::address(data.MiJt_b, world, row, component) * change;
                    velocity += degree * response;
                    if (mass_split) physical_delta += response;
                }
            }
        }
    }
    if (global_dof >= 0) {
        if (mass_split)
            // Algebraically (v_local-v_start)/degree, without subtractive cancellation.
            atomicAdd(wp::address(data.velocity_delta, world, owned + component), physical_delta);
        else
            *wp::address(data.velocity, global_dof) = velocity;
    }
    // A serial-tail pair can transfer a body to different lanes in the next pair.
    __syncwarp(mask);
#endif
""")
def _solve_pair_warp(data: _SolveData, world: int, pair: int, mass_split: int): ...


@wp.kernel(enable_backward=False)
def _solve_color(color: int, data: _SolveData):
    world, index = wp.tid()
    start = data.color_offsets[world, color]
    end = data.color_offsets[world, color + 1]
    if index < end - start:
        _solve_pair(data, world, data.color_pairs[world, start + index])


@wp.kernel(enable_backward=False)
def _solve_tail(color: int, data: _SolveData):
    world = wp.tid()
    for index in range(data.color_offsets[world, color], data.color_offsets[world, color + 1]):
        _solve_pair(data, world, data.color_pairs[world, index])


@wp.kernel(enable_backward=False)
def _solve_color_warp(color: int, data: _SolveData):
    world, index, _lane = wp.tid()
    start = data.color_offsets[world, color]
    end = data.color_offsets[world, color + 1]
    if index < end - start:
        _solve_pair_warp(data, world, data.color_pairs[world, start + index], 0)


@wp.kernel(enable_backward=False)
def _solve_tail_warp(color: int, data: _SolveData):
    world, _lane = wp.tid()
    for index in range(data.color_offsets[world, color], data.color_offsets[world, color + 1]):
        _solve_pair_warp(data, world, data.color_pairs[world, index], 0)


@wp.kernel(enable_backward=False)
def _count_pair_degrees(
    pair_count: wp.array[int],
    pair_a: wp.array2d[int],
    pair_b: wp.array2d[int],
    direct_static_pairs: int,
    degree: wp.array2d[int],
    static_pair_count: wp.array[int],
):
    world = wp.tid()
    static_pairs = int(0)
    for pair in range(pair_count[world]):
        a = pair_a[world, pair]
        b = pair_b[world, pair]
        if a == -1:
            static_pairs += 1
            if direct_static_pairs != 0:
                continue
        if a >= 0:
            degree[world, a] += 1
        if b >= 0 and b != a:
            degree[world, b] += 1
    static_pair_count[world] = static_pairs


@wp.kernel(enable_backward=False)
def _mark_pair_heads(max_rows: int, keys: wp.array[wp.int64], heads: wp.array[int]):
    flat = wp.tid()
    index = flat % max_rows
    key = keys[flat]
    head = int(0)
    if (key & wp.int64(4294967295)) != wp.int64(4294967295):
        if index == 0:
            head = 1
        elif key != keys[flat - 1]:
            head = 1
    heads[flat] = head


@wp.kernel(enable_backward=False)
def _emit_pair_groups(
    max_rows: int,
    keys: wp.array[wp.int64],
    indices: wp.array[int],
    heads: wp.array[int],
    prefix: wp.array[int],
    lengths: wp.array2d[int],
    row_indices: wp.array2d[int],
    row_pair: wp.array2d[int],
    pair_offsets: wp.array2d[int],
    pair_a: wp.array2d[int],
    pair_b: wp.array2d[int],
    unit_count: wp.array[int],
    pair_count: wp.array[int],
    static_pair_count: wp.array[int],
):
    flat = wp.tid()
    world = flat // max_rows
    index = flat % max_rows
    base = world * max_rows
    preceding = int(0)
    if world > 0:
        preceding = prefix[base - 1]
    local_key = keys[flat] & wp.int64(4294967295)
    valid = local_key != wp.int64(4294967295)
    terminal = bool(False)
    if valid:
        pair = prefix[flat] - preceding - 1
        row = indices[flat]
        row_indices[world, index] = row
        for child in range(lengths[world, row]):
            row_pair[world, row + child] = pair
        if heads[flat] != 0:
            pair_offsets[world, pair] = index
            pair_a[world, pair] = int(local_key >> wp.int64(16)) - 1
            pair_b[world, pair] = int(local_key & wp.int64(65535)) - 1
        # Static pairs are a prefix. Only its last unit writes the count.
        if (local_key >> wp.int64(16)) == wp.int64(0):
            last_static = index == max_rows - 1
            if index + 1 < max_rows:
                next_key = keys[flat + 1] & wp.int64(4294967295)
                last_static = (next_key >> wp.int64(16)) != wp.int64(0)
            if last_static:
                static_pair_count[world] = pair + 1
        terminal = index == max_rows - 1
    else:
        if index == 0:
            terminal = True
        elif (keys[flat - 1] & wp.int64(4294967295)) != wp.int64(4294967295):
            terminal = True
    # The first sentinel, or last valid slot of a full world, owns all end values.
    if terminal:
        units = index
        if valid:
            units += 1
        pairs = prefix[flat] - preceding
        unit_count[world] = units
        pair_count[world] = pairs
        pair_offsets[world, pairs] = units


@wp.kernel(enable_backward=False)
def _count_pair_degrees_parallel(
    pair_count: wp.array[int],
    pair_a: wp.array2d[int],
    pair_b: wp.array2d[int],
    direct_static_pairs: int,
    degree: wp.array2d[int],
):
    world, pair = wp.tid()
    if pair >= pair_count[world]:
        return
    a = pair_a[world, pair]
    b = pair_b[world, pair]
    if a == -1 and direct_static_pairs != 0:
        return
    if a >= 0:
        wp.atomic_add(degree, world, a, 1)
    if b >= 0 and b != a:
        wp.atomic_add(degree, world, b, 1)


@wp.kernel(enable_backward=False)
def _prepare_projection_coefficients(
    row_pair: wp.array2d[int],
    pair_a: wp.array2d[int],
    world_dof_indices: wp.array2d[int],
    body_degree: wp.array2d[int],
    meta: wp.array2d[int],
    J_a: wp.array3d[float],
    J_b: wp.array3d[float],
    MiJt_a: wp.array3d[float],
    MiJt_b: wp.array3d[float],
    row_w: wp.array2d[float],
    regularize: int,
    direct_static_pairs: int,
    coefficients: wp.array2d[wp.vec2],
    ready: wp.array[int],
):
    world, row = wp.tid()
    if row == 0:
        # All coefficient writes complete before a subsequent solve launch.
        ready[world] = 1 + regularize
    coefficients[world, row] = wp.vec2(0.0)
    pair = row_pair[world, row]
    if pair < 0:
        return
    effective_inverse = _float_from_bits(meta[world, row * 4 + 1])
    if effective_inverse <= 0.0:
        return
    endpoints = meta[world, row * 4]
    a = _endpoint_a(endpoints)
    b = _endpoint_b(endpoints)
    direct = direct_static_pairs != 0 and pair_a[world, pair] == -1
    extra = float(0.0)
    for component in range(6):
        if a >= 0 and world_dof_indices[world, a + component] >= 0:
            jacobian = J_a[world, row, component]
            response = MiJt_a[world, row, component]
            if a == b:
                jacobian += J_b[world, row, component]
                response += MiJt_b[world, row, component]
            degree = int(1)
            if not direct:
                degree = body_degree[world, a]
            extra += float(degree - 1) * jacobian * response
        if b >= 0 and b != a and world_dof_indices[world, b + component] >= 0:
            degree = int(1)
            if not direct:
                degree = body_degree[world, b]
            extra += float(degree - 1) * J_b[world, row, component] * MiJt_b[world, row, component]
    old_diagonal = 1.0 / effective_inverse
    split_diagonal = old_diagonal + extra
    residual_coefficient = 1.0 / split_diagonal
    impulse_coefficient = float(0.0)
    if regularize != 0 and (meta[world, row * 4 + 3] & 65535) == 0:
        weight = row_w[world, row]
        if weight <= 0.0:
            residual_coefficient = 0.0
            impulse_coefficient = 1.0
        else:
            # Keep g=Dold*(1-w)/w unchanged; avoid division by tiny w.
            weighted_g = old_diagonal * (1.0 - weight)
            denominator = weight * split_diagonal + weighted_g
            residual_coefficient = weight / denominator
            impulse_coefficient = weighted_g / denominator
    coefficients[world, row] = wp.vec2(residual_coefficient, impulse_coefficient)


@wp.kernel(enable_backward=False)
def _solve_mass_split_pairs(data: _SolveData):
    world, index, _lane = wp.tid()
    pair = index
    if data.direct_static_pairs != 0:
        pair += data.static_pair_count[world]
    if pair < data.pair_count[world]:
        _solve_pair_warp(data, world, pair, 1)


@wp.kernel(enable_backward=False)
def _solve_static_pairs(data: _SolveData):
    world, pair, _lane = wp.tid()
    # Canonical (-1, body) keys form a prefix and are mutually body-disjoint.
    if pair < data.static_pair_count[world]:
        _solve_pair_warp(data, world, pair, 0)


@wp.kernel(enable_backward=False)
def _apply_mass_split_delta(data: _SolveData):
    world, dof = wp.tid()
    global_dof = data.world_dof_indices[world, dof]
    if global_dof >= 0:
        data.velocity[global_dof] += data.velocity_delta[world, dof]


class MultiBlockContactSolver:
    """Schedule immediate-response free-body contact pairs, without staging whole worlds.

    Normal rows own their adjacent friction pair. Stable pair-key sorting keeps
    each pair's units in original row order. Pairs within a color are body-disjoint;
    color ordering changes the Gauss-Seidel permutation of dependent pairs.
    Uncolorable pairs run serially after the colors, never dropping constraints.
    This private experimental route supports the current scalar friction law.
    Its caller owns dense rows, velocity-limit tails, iteration counts and fallback.
    CUDA uses one warp/block per pair, with twelve lanes retaining body velocities;
    CPU retains the scalar reference schedule. Experimental ``mass_split`` instead
    solves all pairs from frozen velocities using degree-scaled inverse response,
    then averages their velocity changes. Physical masses and impulses are unchanged.
    The CUDA consensus solve retains scalar Coulomb projection; it does not
    imply a general convergence guarantee for coupled frictional contacts.
    ``direct_static_pairs`` optionally removes static pairs from splitting degrees
    and solves their body-disjoint prefix directly after dynamic consensus.
    ``parallel_prepare`` uses a scan to group pairs without computing unused colors;
    its color diagnostics are unavailable. Both preparation paths support CPU.
    ``cache_coefficients`` prepares immutable row projection coefficients once
    after each pair rebuild; it changes neither the friction law nor sweep count.
    """

    def __init__(
        self,
        world_count,
        max_rows,
        max_world_dofs,
        world_dof_indices,
        device,
        *,
        max_colors=32,
        mode="colored",
        direct_static_pairs=False,
        parallel_prepare=False,
        cache_coefficients=False,
    ):
        if mode not in ("colored", "mass_split"):
            raise ValueError("mode must be 'colored' or 'mass_split'")
        if direct_static_pairs and mode != "mass_split":
            raise ValueError("direct_static_pairs requires mode='mass_split'")
        if parallel_prepare and mode != "mass_split":
            raise ValueError("parallel_prepare requires mode='mass_split'")
        if cache_coefficients and mode != "mass_split":
            raise ValueError("cache_coefficients requires mode='mass_split'")
        if not 1 <= max_colors <= 32:
            raise ValueError("max_colors must be between 1 and 32")
        if not 0 <= world_count < 2147483648 or not 1 <= max_rows <= 65535:
            raise ValueError("World count or row capacity exceeds the packed MF metadata range")
        if not 1 <= max_world_dofs <= 32768:
            raise ValueError("max_world_dofs must fit signed 16-bit MF endpoints")
        self.world_count = int(world_count)
        self.max_rows = int(max_rows)
        self.max_world_dofs = int(max_world_dofs)
        self.max_colors = int(max_colors)
        self.mode = mode
        self.direct_static_pairs = bool(direct_static_pairs)
        self.parallel_prepare = bool(parallel_prepare)
        self.cache_coefficients = bool(cache_coefficients)
        self.colors_available = not self.parallel_prepare
        if self.parallel_prepare and self.world_count * self.max_rows > 2147483647:
            raise ValueError("Parallel preparation requires a signed 32-bit global scan length")
        # Each colored pair consumes at least one six-DOF free body; the one
        # static/static key consumes none. Tail pairs do not use this bound.
        self._pairs_per_color_bound = min(self.max_rows, self.max_world_dofs // 6 + 1)
        bodies = self.max_world_dofs // 6
        # Unordered pairs include self pairs and one additional static endpoint.
        self._pair_bound = min(self.max_rows, bodies * (bodies + 3) // 2 + 1)
        self.device = wp.get_device(device)
        self.world_dof_indices = world_dof_indices
        shape = (self.world_count, self.max_rows)
        self._keys = wp.empty(2 * self.world_count * self.max_rows, dtype=wp.int64, device=self.device)
        self._indices = wp.empty(2 * self.world_count * self.max_rows, dtype=wp.int32, device=self.device)
        self._pair_heads = None
        self._pair_prefix = None
        if self.parallel_prepare:
            self._pair_heads = wp.empty(self.world_count * self.max_rows, dtype=int, device=self.device)
            self._pair_prefix = wp.empty(self.world_count * self.max_rows, dtype=int, device=self.device)
        self.unit_lengths = wp.zeros(shape, dtype=wp.int32, device=self.device)
        self.row_indices = wp.zeros(shape, dtype=wp.int32, device=self.device)
        self.row_pair = wp.full(shape, -1, dtype=wp.int32, device=self.device)
        self.pair_offsets = wp.zeros((self.world_count, self.max_rows + 1), dtype=wp.int32, device=self.device)
        self.pair_a = wp.zeros(shape, dtype=wp.int32, device=self.device)
        self.pair_b = wp.zeros(shape, dtype=wp.int32, device=self.device)
        self.pair_color = wp.full(shape, -1 if self.parallel_prepare else 0, dtype=wp.int32, device=self.device)
        self.pair_count = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.unit_count = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.color_count = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.color_counts = wp.zeros((self.world_count, self.max_colors), dtype=wp.int32, device=self.device)
        self.tail_count = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.color_offsets = wp.zeros((self.world_count, self.max_colors + 2), dtype=wp.int32, device=self.device)
        self.color_pairs = wp.zeros(shape, dtype=wp.int32, device=self.device)
        self._cursors = wp.zeros((self.world_count, self.max_colors + 1), dtype=wp.int32, device=self.device)
        self._body_masks = wp.zeros((self.world_count, self.max_world_dofs), dtype=wp.uint32, device=self.device)
        self.body_degree = None
        self.static_pair_count = None
        self.velocity_delta = None
        self.coefficients = None
        self.coefficient_ready = None
        self.coefficient_cache_misses = None
        self._topology_prepared = False
        self._coefficient_inputs = None
        self._data = _SolveData()
        self._data.direct_static_pairs = int(self.direct_static_pairs)
        for name in (
            "world_dof_indices",
            "row_indices",
            "unit_lengths",
            "pair_offsets",
            "pair_a",
            "pair_b",
            "pair_count",
            "color_offsets",
            "color_pairs",
        ):
            setattr(self._data, name, getattr(self, name))
        if mode == "mass_split":
            dof_shape = (self.world_count, self.max_world_dofs)
            self.body_degree = wp.zeros(dof_shape, dtype=int, device=self.device)
            self.static_pair_count = wp.zeros(self.world_count, dtype=int, device=self.device)
            self.velocity_delta = wp.zeros(dof_shape, dtype=float, device=self.device)
            self._data.body_degree = self.body_degree
            self._data.static_pair_count = self.static_pair_count
            self._data.velocity_delta = self.velocity_delta
        if self.cache_coefficients:
            self.coefficients = wp.zeros(shape, dtype=wp.vec2, device=self.device)
            self.coefficient_ready = wp.zeros(self.world_count, dtype=int, device=self.device)
            self.coefficient_cache_misses = wp.zeros(self.world_count, dtype=int, device=self.device)
            self._data.coefficients = self.coefficients
            self._data.coefficient_ready = self.coefficient_ready
            self._data.coefficient_cache_misses = self.coefficient_cache_misses

    def prepare(self, mf_constraint_count, mf_contact_rows_end, mf_meta):
        """Group current contact units on-device, optionally building a color schedule."""
        self._topology_prepared = True
        self._coefficient_inputs = None
        if self.coefficient_ready is not None:
            self.coefficient_ready.zero_()
        if not self.world_count:
            return
        if not self.parallel_prepare:
            self._body_masks.zero_()
        count = self.world_count * self.max_rows
        wp.launch(
            _make_keys,
            dim=count,
            inputs=[self.max_rows, mf_constraint_count, mf_contact_rows_end, mf_meta],
            outputs=[self._keys, self._indices, self.unit_lengths, self.row_pair],
            device=self.device,
        )

        wp.utils.radix_sort_pairs(self._keys, self._indices, count)
        if self.parallel_prepare:
            wp.launch(
                _mark_pair_heads,
                dim=count,
                inputs=[self.max_rows, self._keys],
                outputs=[self._pair_heads],
                device=self.device,
            )
            wp.utils.array_scan(self._pair_heads, self._pair_prefix, inclusive=True)
            self.static_pair_count.zero_()
            wp.launch(
                _emit_pair_groups,
                dim=count,
                inputs=[
                    self.max_rows,
                    self._keys,
                    self._indices,
                    self._pair_heads,
                    self._pair_prefix,
                    self.unit_lengths,
                ],
                outputs=[
                    self.row_indices,
                    self.row_pair,
                    self.pair_offsets,
                    self.pair_a,
                    self.pair_b,
                    self.unit_count,
                    self.pair_count,
                    self.static_pair_count,
                ],
                device=self.device,
            )
            self.body_degree.zero_()
            wp.launch(
                _count_pair_degrees_parallel,
                dim=(self.world_count, self._pair_bound),
                inputs=[self.pair_count, self.pair_a, self.pair_b, int(self.direct_static_pairs)],
                outputs=[self.body_degree],
                device=self.device,
            )
            return
        wp.launch(
            _group_and_color,
            dim=self.world_count,
            inputs=[self.max_rows, self.max_colors, self._keys, self._indices, self.unit_lengths],
            outputs=[
                self._body_masks,
                self.row_indices,
                self.row_pair,
                self.pair_offsets,
                self.pair_a,
                self.pair_b,
                self.pair_color,
                self.unit_count,
                self.pair_count,
                self.color_count,
                self.color_counts,
                self.tail_count,
                self.color_offsets,
                self.color_pairs,
                self._cursors,
            ],
            device=self.device,
        )
        if self.mode == "mass_split":
            self.body_degree.zero_()
            wp.launch(
                _count_pair_degrees,
                dim=self.world_count,
                inputs=[self.pair_count, self.pair_a, self.pair_b, int(self.direct_static_pairs)],
                outputs=[self.body_degree, self.static_pair_count],
                device=self.device,
            )

    def prepare_coefficients(self, mf_meta, mf_J_a, mf_J_b, mf_MiJt_a, mf_MiJt_b, mf_row_w, regularize):
        """Cache delta=-a*residual-b*lambda after current pairs and responses are ready.

        Call once after ``prepare`` and final J/response writes, before the
        sweep loop. Include both calls in captured graphs that rebuild rows.
        A replay missing coefficient refresh uses the uncached projector and
        sets ``coefficient_cache_misses`` rather than consuming stale values.
        """
        if not self.cache_coefficients or self.coefficients is None:
            raise ValueError("Projection caching requires cache_coefficients=True at construction")
        if not self._topology_prepared:
            raise RuntimeError("Call prepare before preparing projection coefficients")
        regularize = int(bool(regularize))
        operands = (mf_meta, mf_J_a, mf_J_b, mf_MiJt_a, mf_MiJt_b, mf_row_w)
        self._coefficient_inputs = (*(array.ptr for array in operands), regularize, self.direct_static_pairs)
        if not self.world_count:
            return
        wp.launch(
            _prepare_projection_coefficients,
            dim=(self.world_count, self.max_rows),
            inputs=[
                self.row_pair,
                self.pair_a,
                self.world_dof_indices,
                self.body_degree,
                *operands,
                regularize,
                int(self.direct_static_pairs),
            ],
            outputs=[self.coefficients, self.coefficient_ready],
            device=self.device,
        )

    def _bind(
        self,
        mf_meta,
        mf_impulses,
        mf_J_a,
        mf_J_b,
        mf_MiJt_a,
        mf_MiJt_b,
        mf_row_mu,
        mf_row_w,
        v_out,
        omega,
        regularize,
        friction_active,
    ):
        data = self._data
        data.meta = mf_meta
        data.impulses = mf_impulses
        data.J_a = mf_J_a
        data.J_b = mf_J_b
        data.MiJt_a = mf_MiJt_a
        data.MiJt_b = mf_MiJt_b
        data.mu = mf_row_mu
        data.row_w = mf_row_w
        data.velocity = v_out
        data.omega = float(omega)
        data.regularize = int(regularize)
        data.friction_active = int(friction_active)
        data.cache_coefficients = int(self.cache_coefficients)
        return data

    def solve_color(
        self,
        color,
        mf_meta,
        mf_impulses,
        mf_J_a,
        mf_J_b,
        mf_MiJt_a,
        mf_MiJt_b,
        mf_row_mu,
        mf_row_w,
        v_out,
        omega,
        regularize,
        friction_active,
    ):
        """Solve one conflict-free pair color across all worlds and CUDA blocks."""
        if not self.colors_available:
            raise ValueError("Parallel mass-split preparation does not compute a color schedule")
        if not 0 <= color < self.max_colors:
            raise ValueError("color must be below max_colors")
        if not self.world_count:
            return
        data = self._bind(
            mf_meta,
            mf_impulses,
            mf_J_a,
            mf_J_b,
            mf_MiJt_a,
            mf_MiJt_b,
            mf_row_mu,
            mf_row_w,
            v_out,
            omega,
            regularize,
            friction_active,
        )
        wp.launch(
            _solve_color_warp if self.device.is_cuda else _solve_color,
            dim=(self.world_count, self._pairs_per_color_bound, 32)
            if self.device.is_cuda
            else (self.world_count, self._pairs_per_color_bound),
            inputs=[color, data],
            block_dim=32,
            device=self.device,
        )

    def solve_tail(
        self,
        mf_meta,
        mf_impulses,
        mf_J_a,
        mf_J_b,
        mf_MiJt_a,
        mf_MiJt_b,
        mf_row_mu,
        mf_row_w,
        v_out,
        omega,
        regularize,
        friction_active,
    ):
        """Solve every color-overflow pair serially per world after the parallel colors."""
        if not self.colors_available:
            raise ValueError("Parallel mass-split preparation does not compute a color schedule")
        if not self.world_count:
            return
        data = self._bind(
            mf_meta,
            mf_impulses,
            mf_J_a,
            mf_J_b,
            mf_MiJt_a,
            mf_MiJt_b,
            mf_row_mu,
            mf_row_w,
            v_out,
            omega,
            regularize,
            friction_active,
        )
        wp.launch(
            _solve_tail_warp if self.device.is_cuda else _solve_tail,
            dim=(self.world_count, 32) if self.device.is_cuda else self.world_count,
            inputs=[self.max_colors, data],
            block_dim=32,
            device=self.device,
        )

    def solve_mass_split(
        self,
        mf_meta,
        mf_impulses,
        mf_J_a,
        mf_J_b,
        mf_MiJt_a,
        mf_MiJt_b,
        mf_row_mu,
        mf_row_w,
        v_out,
        omega,
        regularize,
        friction_active,
    ):
        """Perform one simultaneous pair sweep and consensus update, without changing masses."""
        if self.mode != "mass_split":
            raise ValueError("solve_mass_split requires mode='mass_split'")
        if self.cache_coefficients:
            operands = (mf_meta, mf_J_a, mf_J_b, mf_MiJt_a, mf_MiJt_b, mf_row_w)
            signature = (*(array.ptr for array in operands), int(bool(regularize)), self.direct_static_pairs)
            if self._coefficient_inputs != signature:
                raise RuntimeError("Refresh projection coefficients after prepare or changing coefficient inputs")
        if not self.device.is_cuda:
            raise ValueError("Experimental mass_split solve requires a CUDA device")
        if not self.world_count:
            return
        data = self._bind(
            mf_meta,
            mf_impulses,
            mf_J_a,
            mf_J_b,
            mf_MiJt_a,
            mf_MiJt_b,
            mf_row_mu,
            mf_row_w,
            v_out,
            omega,
            regularize,
            friction_active,
        )
        self.velocity_delta.zero_()
        wp.launch(
            _solve_mass_split_pairs,
            dim=(self.world_count, self._pair_bound, 32),
            inputs=[data],
            block_dim=32,
            device=self.device,
        )
        wp.launch(
            _apply_mass_split_delta,
            dim=(self.world_count, self.max_world_dofs),
            inputs=[data],
            device=self.device,
        )
        if self.direct_static_pairs:
            wp.launch(
                _solve_static_pairs,
                dim=(self.world_count, self._pairs_per_color_bound, 32),
                inputs=[data],
                block_dim=32,
                device=self.device,
            )
