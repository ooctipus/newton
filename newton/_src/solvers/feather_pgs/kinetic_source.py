# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check retained internal source definitions at import/factory time only.

These guards cover the selected numerical definitions, not unrelated Solver
methods or an entire edited source file. Native string values remain exact.
The cache is process-local; this is source-construction admission, not a hot-loop
file watcher. No scratch path, public API, or third-party parser is introduced.
"""

import ast
import copy
import functools
import hashlib
from pathlib import Path

PINNED = {
    "contact_allocation_source.py": {
        "_BODIES": "b5476950d4d9abe81ea03201ffe8ac92ad46c5bc72e66a9fd85c104138e1db6b",
        "_WORLD": "f4d0d3bb212a2554312f3c6c100461fe6dd9f256d5a182c8b34014c9367020c1",
        "_PHI": "0075baa843e40e3c7c0ec4ddd26b14b397c47a73f011c2d414e403a275166c6a",
        "_RESOLVED": "54b6f3f3cd95fea186492a4144c1ce41aabebef2bf77e820afeefa580e6c95de",
        "_HELPER": "adb4b579ac2cb68406804ce53256dcccabf5aaa18ffa0942551b050096f08cf4",
        "_KERNEL": "55c3009578f93abf574ab1f76474eabc3e0cc2a7837b99eda5b31c486d216872",
        "get_source": "36587caa26a12c4fa2f223071c5eee3e44decb186defab603ddab1951c5ca38c",
        "bind": "2459bd92c84b87e6b3b24ca3b10ba242ebde853bbc09f9a66bf41c76fd426c01",
    },
    "mf_dispatch.py": {
        "_NATIVE_ARGUMENTS": "795a55f6b5227fce9bba9d2df228dc0560ba344eb9e4da0227cb17684d59d28f",
        "_QUEUE_ARGUMENTS": "a3dce18c710b83d8747e3b4306183bcdf73a1cccf5c03913816ae1492dc9e947",
        "_compile": "ddcb24a52a1cf1d26c9609c818a5eaf69d480eff7374b640a1c0dbdddd697d43",
        "_signature": "902b3486f6499d0bad23cc6bb54c50aa6705ff9155c1f5493667e071c496ce81",
        "make_native": "a5d5100996cc1be1d8af2265552cfce787bdfa14c75409faf59f6c31324debfa",
        "make_dispatch": "d841e99fb7900119d4a6f2d02b01c86935f9f5ab3d7bed735351e388e217659a",
    },
    "paired_factor_source.py": {
        "_physical_entry": "57b685e553d3b34f3a7958019b33e39d48308ac71faefeaddcda895f48f65bfb",
        "_physical_exit": "ee488b1935af525104f25aca4904ef3b26409ba9ec19baff61edef6d71966101",
        "get_source": "0014db0213be46f8f9dc7f4981d72af856f867022588cee986a0e65a73fef3af",
    },
    "small_cholesky.py": {
        "_FACTOR": "0622916ba6629039cec48620ddcc0bb3b8208f7ba4ce2e7af8c9c4dce04a2134",
        "_FORCE_INIT": "4e1883323ae63f673815d3f0e6aa1ef93365e126a1b6817f4cb2986e3c4faa3c",
        "_COMPOSITE_INPUT": "1ec974e47efd2095efa91bca708d53d96e78051dff51faddb44159efd9ba5b9d",
        "_INPUTS": "10b6aa85102d13efe7831ef5742108f1c689e04542dbb6bedcf9f3340e5f3ed4",
        "get_kernel": "d1cd133efac455c6e9f59821af782fec4f2ff3b6f9102dfb5246000f19299204",
    },
    "solver_feather_pgs.py": {
        "_get_pgs_solve_paired_factor_kernel": "1cc5dd5422f40ed122d0dce461553bbe8fbdd73fb5df84d59e7545ab97c1590a",
        "_get_pgs_solve_mf_gs_kernel": "35602b2592d4131f24f31518c17ccd5275684b81502311ee2faaabf1e10d9dd5",
        "_get_crba_cholesky_warp_kernel": "f507209b4a71155f75aa86a783241723e1370c41a0c262744c020bfc224969de",
        "_get_inverse_cholesky_register_kernel": "f1f2c60767d69d8006fb074c768877b921579954b7d36bc1efee9366212234d3",
        "_get_cholesky_kernel": "c262785fdd51c25f94c3c67ab4bb4cbe0fa95f16b61adcb788fbd16de10a4689",
        "_get_pack_mf_meta_kernel": "353265897c2af4cb85fc0e284b25bb9a862f67b54b74edd2e7b472a331c55a81",
    },
    "kernels.py": {
        "_contact_allocator": "691354a0e712447d8ed88bb4021cc75b720b2c9a597689b4661730b8423a1a09",
        "_allocate_world_contact_slot": "0e908992c5b8a236538cd9f07c7d92a4205df1d5e07fedcf81f51699285d7722",
        "allocate_world_contact_slots": "fc7bd52682bb252393da31ce2bc307bd40ce5a9db13101b4c3dad6171aeda789",
        "allocate_rigid_velocity_limit_slots": "0c85ccc6a10fb4a59099870574929fe80c8a8b21480a3d91d02f6671c2d195fa",
        "finalize_constraint_counts_with_status": "9d45d0c596ab5919f1e7a40882cbbaeceb693db358c800db67835aef8ba3b3e0",
        "compute_contact_linear_force_from_impulses": "222aca7ac6ffa8ced2cb523b77ee4d2b88bb9f95fedebae17b616b1245b4791a",
        "_contact_linear_force": "4556ff3c816a41bca297bf26eb3589161f8976d1aadbaf0f78b11eda1f4f8864",
        "pack_contact_linear_force_as_spatial": "df439829fe0c8dfbc48961bed73f1a6917e10865ac21bd57cd6ec5d1d89185f1",
        "_pack_contact_linear_force": "75e0c5272892943a4c0f4012ec1613513c0ed1c5a7d82cde0e9942f411b06e8b",
        "compute_mf_body_Hinv": "90c5abc526935ac81ad71dae92eb2ed16ff5820ab9e7bda0fdb6a52136a0a9f7",
        "build_mf_contact_rows": "8520c549eb3fbfa7ab9c41fa8b8053cba3669c10baa6b3d96b48c51beb0a4480",
        "populate_rigid_velocity_limit_rows": "d71b12578c40b4c805740822c8b4de0e2b231cfeeb4a54910171fd937f1982cf",
        "compute_mf_world_dof_offsets": "ebfa6171449f73873897da65972c2e1f01bc707386733a27121a086081f9dccc",
        "compute_mf_effective_mass_and_rhs": "1e819d242097c4722c3db16512c7c9838d86dbc2a3b83f63f7dfdf391694c641",
        "update_qdd_from_velocity": "96efbd84731f509e3472bb3446fa3088583b4f29bbf27aad2df03a5a12a64ef9",
        "remove_free_root_transport_from_qdd": "bfd6fc295724f3ef7f48070251b4001d72286ce78bdf0714ab485a4cea780bd0",
        "integrate_generalized_joints": "f8018cdc85df8944041886f723b299fe6a50f24fea9a730308fc5675420a4e58",
        "jcalc_transform": "84d09399dec35112103e5b4bf315e81eddf5199a65075219d66a1f03df7e0996",
        "jcalc_motion": "15c927d5066b05e650fb66f1a28ccefbe8288218caba86a57d06da21e13bf608",
        "finalize_body_dynamics_values": "eb0862fab868f5d0bfcfcf3b78161fc4b96cc23a35744ecd8439e6e0fb9867ff",
        "finalize_body_dynamics_body": "16a2637f627c9b13948079cc96e8bc6d09fb7958f49f3f34c0a9a77672823d63",
        "apply_free_root_velocity_corrections": "4d3cf47567345444363054a392cfc1676663f35e7f86cd30ce8fb0c42657c09f",
    },
    "cached_state.py": {
        "_INTEGRATION": "33f3c5af040aa39de98421346673484f257af0bd542d68b4098137e204c65457",
        "_RELATIVE": "7f1820f6cfaabfdf6ddaf55726577cc2246e01d41022daf1159f74ef1837a8f0",
        "_BODY_TERMS": "ef664e8ed9bf3772cecfcdcec9e58ebcc92c7a11cbd4d33c32b15d5356622ed4",
        "generalized_operations": "7bd9892a8e09ca51b5775d7552c7f6ef1b3e0d38d99d5242ec36f07f0a733fbc",
        "make_state_kernel": "8332bec15de106b47a9b5e3b8ded7bc988d70507a2ed4901c7565a1e57f08274",
        "make_body_terms": "00234de7e281fefae0c7c9830117848a1ec0c25466bbd1a767c455e0b9a1bba0",
    },
    "cached_state_forest.py": {
        "_RELATIVE": "b291f1964566690e5bc7ad5fc7f79a1352c0cdbb227760fff3ffe1fec303b500",
        "_ORIGINS": "13c17885220eeebd65f6212f43e76c23c7ae5a019ba024e25be5dbc674e282bf",
        "_MOTION": "5ea5b1019cb1d988f4fd3dd05137f70a3074cc9ffcef574d73791a214c29e2b8",
        "_BODIES": "792cc984b7c272c708953feeb47a8780492f88e7f695ae07ab6286c75e6d87ef",
        "_COLLECT": "476792fbbf181045c9ec22114b77221fd0eb9f43ae6038f19a05cb53011ee656",
        "make_forest_state": "be5424d94f2586b384d07768f3fa70b8d58e50b8b7967896cd64ffecc372aa22",
    },
    "mf_gs_source.py": {
        "GeneratedSource": "e7ab064517dd2cdfee617fb913870e0bb880abfc0836853b58ca3a5928f13447",
        "get_source": "028d9b1171cbaa282216416c24e66a5b7c2ac587a1afd9879cd06a0316dd81d3",
    },
    "row_metadata.py": {
        "_DENSE_META_ROW_TYPE_BITS": "79f563eb47c41c43a1cec63cfe314f58b65e63352b7bc4fcec1e786dc52db61e",
        "_DENSE_META_ROW_TYPE_MASK": "1fb8afd3532aa05d0456dfc05907c938e658ac18978f196b3a18c25735fae445",
        "_DENSE_META_MAX_PARENT": "f0353f388c94e472cc50bbd521a7881bdae80811fcbed5ef678cf25d374078a7",
        "_MF_META_MAX_CAPACITY": "c0469c6b148f38bd136f0d1ccda93d11a9e8145a9d0daecd42a4e0ea32c734b2",
        "_validate_dense_metadata_encoding": "cea9368fd58e2b11dc023931bf20d59ed17e512bd07213bf4377bdb7d45f42e7",
        "_validate_mf_metadata_encoding": "5cba2a7fd0f6f44fe2fb7bd0a39611d4b65560a746c83cbcba36a51083fb7b42",
        "_mf_parent_expression": "183030b9f3be65dccaeab30bf33698c73c5cc65ef1f34a124ec72ff7ae7b3a62",
    },
    "friction.py": {
        "_PAIR_METRIC": "6c95081e6eade8668d063255732c8a07699f06b58cd7e5cf09d8d65ff992c380",
        "_PAIR_STATE": "85834499a0c68b509e6067bdfefd201214a28b89215002c7fbf00a014ca33f11",
        "_PAIR_ENTRY": "cd37230c9ccc55c127edbed5c7eff31a2201b8e1e49143f3cf14be1769298928",
        "get_pair_source": "3f987616ed963d7f1328bb651454a2316d4a34daa767e6eb955cae115f9c467a",
    },
}


class _ArrayAnnotations(ast.NodeTransformer):
    """Normalize only equivalent Warp array type declarations."""

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "wp"
            and node.func.attr in ("array", "array2d", "array3d", "array4d")
            and not node.args
            and len(node.keywords) == 1
            and node.keywords[0].arg == "dtype"
        ):
            return ast.Subscript(value=node.func, slice=node.keywords[0].value, ctx=ast.Load())
        return node


def definition_digest(node):
    """Hash a definition without location/formatting or array spelling noise."""
    normalized = _ArrayAnnotations().visit(copy.deepcopy(node))
    return hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest()


def check_source(source, expected):
    """Validate selected definitions; leave unrelated methods outside this guard."""
    definitions = {}
    for node in ast.parse(source).body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        for name in names:
            if name in definitions:
                raise RuntimeError("Duplicate retained definition: " + name)
            definitions[name] = node
    for name, pin in expected.items():
        if name not in definitions or definition_digest(definitions[name]) != pin:
            raise RuntimeError("Retained kinetic source definition changed: " + name)
    return {name: definitions[name] for name in expected}


@functools.cache
def _checked_definitions(filename, names):
    if filename not in PINNED or any(name not in PINNED[filename] for name in names):
        raise ValueError("Unknown retained kinetic source definition")
    source = Path(__file__).with_name(filename).read_text()
    return check_source(source, {name: PINNED[filename][name] for name in names})


def checked_definitions(filename, names):
    """Return private AST copies; never let an adapter mutate the cached authority."""
    return copy.deepcopy(_checked_definitions(filename, tuple(names)))
