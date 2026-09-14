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
    "solver_feather_pgs.py": {
        "_get_pgs_solve_paired_factor_kernel": "b7c6208114a7d3c8ec3f2eede39e957491fd9396f91bbbcd785923b2c9a2933c",
        "_get_pgs_solve_mf_gs_kernel": "4bc947b5e582383a0ba1fe6085fbb02babe043a42c7021dbcf12eaebac8aeee8",
        "_get_crba_cholesky_warp_kernel": "d55ef23699f3c177c5bfc3da86397487de9014244f19a70fbcfe94f26dad17d6",
        "_get_inverse_cholesky_register_kernel": "f1f2c60767d69d8006fb074c768877b921579954b7d36bc1efee9366212234d3",
        "_get_cholesky_kernel": "c262785fdd51c25f94c3c67ab4bb4cbe0fa95f16b61adcb788fbd16de10a4689",
        "_get_pack_mf_meta_kernel": "0c3e9cc73a403a3eb03673c088e6a1ad7de8d6c871d5b7b125c891e350c0d157",
    },
    "independent_components.py": {
        "get_prepare_kernel": "32e14002b1ebaa489d7560161d0a2f58a810269d23ea653c8ac52aa7d207da4e"
    },
    "kernels.py": {
        "_allocate_world_contact_slot": "fa5e4287b56b768dc3d95bd700c696ac4fd4f1dbc92fc9ca688f00ede5541638",
        "allocate_world_contact_slots": "7a6cc9035d35246d9b3c0620ed0d6fa68df6ddf129102479f3a13f86b2c62721",
        "allocate_rigid_velocity_limit_slots": "0c85ccc6a10fb4a59099870574929fe80c8a8b21480a3d91d02f6671c2d195fa",
        "finalize_constraint_counts_with_status": "faff737dcff3a438a4b575d5fe5136e7b2c4055f09752ebc28ea3b3f8085d8a4",
        "compute_contact_linear_force_from_impulses": "c5201c74b29cd995552f2b2b4e75493066c674db77f86f6a58ca19c0a7b5fed1",
        "pack_contact_linear_force_as_spatial": "c33cdd1013b621ce6f50b91d2f758f0939316a9b8012aa0a653ccb5623d3fdbc",
        "compute_mf_body_Hinv": "90c5abc526935ac81ad71dae92eb2ed16ff5820ab9e7bda0fdb6a52136a0a9f7",
        "build_mf_contact_rows": "0d9c180bede8c31cc8038bd47540d47a172c6c5f3cc52df4eeffbed1a80c2501",
        "populate_rigid_velocity_limit_rows": "d71b12578c40b4c805740822c8b4de0e2b231cfeeb4a54910171fd937f1982cf",
        "compute_mf_world_dof_offsets": "ebfa6171449f73873897da65972c2e1f01bc707386733a27121a086081f9dccc",
        "compute_mf_effective_mass_and_rhs": "34465acd90d4f98b327b7c713dcf47141ab46b554a3e84fd6d1d1e9572974997",
        "update_qdd_from_velocity": "d27c217ecec55796dea417dc5ec87ee576050e0a72ab2455aaca546ae293fb6c",
        "remove_free_root_transport_from_qdd": "ad0c81fd9713f999e434a13dd43bcb396f733c94cb655ef94f4f72c5023c6f21",
        "integrate_generalized_joints": "2418badf5019fb3b34eb3042e57835bda787a238c5227ff857ed950bed2d3a66",
        "jcalc_transform": "84d09399dec35112103e5b4bf315e81eddf5199a65075219d66a1f03df7e0996",
        "jcalc_motion": "15c927d5066b05e650fb66f1a28ccefbe8288218caba86a57d06da21e13bf608",
        "finalize_body_dynamics_body": "fada293f8bbe51006f1b426b66658e6d7f0a9966b72667349c41714108c48657",
    },
    "world_scan_publication.py": {
        "PublicationData": "fcf16576d1e626ccdb95ea13d55bbe84ec55a221d50705f1b03179e5f62d8452",
        "_POSE_SCAN": "bc948784b3102bfcbd74e5fd5d662cbcd5d9ab22f74ed28aa51778befb73bf69",
        "_MOTION_SCAN": "e5229c09e4ce980400c34e2a652be97ec4f4225f088ef86a332596425f1cc1db",
        "_release": "58af3211560e7d4d26ca0b40e25d635e5ede62aec31568d600430a159fcb2715",
        "_sync": "7d41b3345db71397d0e001eaa2c2829f9c1dd3971b55b8e44b8400c0803ff55e",
        "_lane": "2612cc789289f0770732f9d03fb5575e8f29004133291745a61b5ab210c63636",
        "_store_pose": "7883fd68a1d363364aa1d7fcad18432c9ddd8bd4e89eb2158d5e216374208572",
        "_load_pose": "8a83312b137b3161e1157910a16d33fe170659ee92b8b4062a80d525bce8a33c",
        "_store_motion": "87534da0d341eeaec4396d2ee7ff5f250f6a37929da3c3cebb009fbd1c540ec9",
        "_load_motion": "27f387f09a9f8512395577c4e36921c2a75f282c47fdbbe24897919941fd7587",
        "_store_origin": "c908659ca851c7216153bcd9638c2beba96599f68917e9748950bf74f0dd7d85",
        "_load_origin": "13b5f59c633411c9a7bf7caeda8d6fbb4eb6504019029b4d6179cb52374da5a5",
    },
    "kuka_joint_world.py": {
        "JointWorldPlan": "270ea3dbfb1d19f8cea4b2d96eb543ba7337b08c7f49c4018bd75a274d8c6a8d",
        "_TWISTS": "bc6b882ec3b4a74aa4697ee282e1ca7fc31a0cb88946660ca6885acc4514d2e4",
        "_LIMITS": "1a402624e4030408efd39a7a1db85e03e08d91414381dc321c639cd9b62f7011",
        "_STORAGE": "99b0ce220a59a611ca8d8aa8ed6724d2d3a3bcc802e3408b6aa533600ceb9a89",
        "_limits": "ea79facce6db721b0beeaa40d1ced586fbac97a6fdd44ba92e548f1aa00f671a",
        "_ready": "dc63a432de2234472abe307c85ed98ad507381ad7baba67041c1fa082c2df6f0",
        "_storage": "ff666f78b27108b1285331fc44ba61b913e954aa0af9c0758525c30de9367db5",
        "_release": "58af3211560e7d4d26ca0b40e25d635e5ede62aec31568d600430a159fcb2715",
        "_lane": "07615c6d1dd5c43e7ad93a39b56a48e25b87498da0fb5c66403f5910c4b74745",
        "_body_twist": "5837c1ce0a3f402897e71aff09076f14880e2422111160df4264fa9ddef334be",
        "_body_valid": "be8cc8904be29e7a421895f91a51fa463612c244e00bd33a4481730683e7de71",
        "normal_source": "039367da360aa4f9da3338a2c101156b9f327dbd01e0e3cafb1e25e9637f61b2",
        "operation_source": "0ea35321bbd3b2f7ed8dac4c7a8edd04b7004887cf7f51aa55c23cd6f1bb0240",
    },
    "simple_world.py": {
        "_SimpleWorldInput": "041b2334d5fd449d41a0d5f560b101942a4fd6932b8b38f9e07440a310c792ec",
        "_SimpleRawContacts": "93968df80ab45e14da99a8f1f5ab946b9214c9c2117f963102416491918f9c29",
        "_check_normal": "2313727eeb147afcd77733d3e90ba345bf6bce0a2a93e14d9e9ed4a376fcd4e5",
        "_endpoint_speed": "66ab54b792a395c3543450ac37a036563a419a045adc35c8bcc6d67f9d0cffe3",
        "_finite3": "124a1b71c2f27d2e97d8261b2d8efda24591636e76f3607eb0d43b2ee934b227",
        "_finite6": "18d59e5041cf29be3fdf8bd1912bbf87bb7ac25227201ab347bd546dc7bfb6f7",
        "_passes": "45dffb35395ccfff2fefc32c3ec08245f40caacc0778e25aef7d588b81c7c8bd",
    },
    "raw_world_contacts.py": {
        "RawWorldContacts": "6a19f35b7f14600ccf9c56a821d1cc43a832361d5a57dee9ccdb915daba2c6c4",
        "RawWorldContactBuckets": "76c4842fd2fbe4ef01a24aef762349aa2bcaa77ee36d7257018555f180814a02",
        "_endpoint_world": "2ef5ca46a8cb9b1ebdc2d9e18fbaffa348fe14608be46a05c6f0435743920e7d",
        "_contact_world": "9f732181c823800550239c16d3ee4cba2bc6631818dfe572ecdb909d3c1e495f",
        "count_raw_world_contacts": "5efdaa5d8d37903439ae92d6f7d172528b083d025ecddb5d72c485893a216dc5",
        "scatter_raw_world_contacts": "f7998b89a0821e00be34ddebac3bd50bcc38680fd1f3f81d3966beb06f7e6b85",
        "validate_raw_world_contacts": "a26ca8b7ba1500e57584909ed34b1ec6e2d6d9e14f07351fea59e7073a1a4eb8",
    },
    "kinetic_rows.py": {
        "_CONTACT": "714f5417831f89d2cd36413150e95156620ba7e5b78fedbab8e48210cef1f5c3",
        "_t": "bac2d4805d420caba82de9e39ed1d5a3c33014f5c4fbbab9ce2d26bf6ade53ec",
        "_lane": "2612cc789289f0770732f9d03fb5575e8f29004133291745a61b5ab210c63636",
        "_release": "58af3211560e7d4d26ca0b40e25d635e5ede62aec31568d600430a159fcb2715",
    },
    "kinetic_state.py": {
        "_get_kernel": "7fc019e179269f7b16f819f0e3fd4fc74d2eca3f29ed956d312e15f0a9d22f9e",
        "_world_gravity": "faffbf54ec341a09a0d99b459e6efe0956cc7b19026f6574c1bd3c99dc89806f",
        "_admit": "dad7435af0211e5b06822534a1003187a90f54ea997e4b20d5a52e123c85fe87",
        "_collect": "2c197cfa06bc5bbfeff5aa381ab6a7f594f61642a8b9bda765c4174e7dc1d6f6",
        "_COLLECT": "8409f401ce8985e5990e322896d36a57adee6185c5fb1f26c4f58ec7030e2b9b",
        "_primary_terms": "d5f756e1fbc8dfb12015ab0315a2aa74e11af9ff25c5f33402dbda939c78bba8",
    },
    "kinetic_solve.py": {
        "_T": "03521fc92b11e045c58334667f9d55afbb9bd5b353ff4f5ecc0607def94db291",
        "_GUARD": "d05f161e7bd5d18f9a6a7de858fc6a1a1b77889b64498012a73bcc6ecf9d1cc0",
        "_CPU": "2f9d92a80124b9934f7e3f60ad5e92981f8ae2f00738919ef7776b28289260ce",
        "_MATERIALIZE": "b76940135f6894f838e35be11ca2a88abc8365d1d610e49e5f29d3bb77eff2e3",
        "_impulse_transaction_edits": "6eb44c5215d4b478db98e9e427526ca9584514bb03a3160420e2c97be11864a3",
        "recover_impulse_transactions": "3ba7137b8926c79662569883af2fae919af7dd9cf6fee9a6c2065e982021dd7a",
        "repair_impulse_transactions": "35fcc73dac25cabdd80259d4a76e2ecea2f08e7ec980822be8f18f8b893aed15",
        "cuda_recurrence": "0f2c1219b0b4757ff80c2488a4f56d5142b86f1adae06f721a489141946e156f",
        "qualifier_source": "f6675585c6856aae3d5e6e5bb11f89e486705c41b308585c2769a8c6bfa87940",
        "get_solve_kernel": "41401d6224f9e227e9509ff5f9b6343c1e7a3df33bf63b7590fecad5d0ca492e",
        "get_qualify_kernel": "e56490f1a74e258497378048176ee62dc658815ac0e5ad9a915cfad0ec3d04cb",
        "get_materialize_kernel": "60d020f8de5d16ebca4fc0fa1f22e073efd78ddda9df5247cb94d8b0f76b6717",
    },
    "kinetic_hybrid.py": {
        "HybridCoupledData": "1a261fa76dc5207521c424ac6489a29c87c9b6a0448add39538ea356af28a078",
        "_CPU": "f3b8d0e5be27c4f2d58665e74b83e9ff21b37054d777346118ac8235f717e233",
        "_STAGE": "d16346c926d401515e157795d0bd348e8f8114b635d6fe0b35b5e2739c83a4e0",
        "_STORE": "7abc04f9023ba89d1a529090ce8130f89a534ba0f577464bb0c5fc2c5c9c91ec",
        "qualify_source": "cca6d14635c015abab271628cb6cb6a5c0b7ccc5c15c7f616b266f66d26ec1aa",
        "general_source": "1647884f80ae7da11bf1dd2fa29cef11835d6b825062cad4328e3651a8066b85",
        "get_qualify_kernel": "f524aab764befe22c841faf9b1b0e52a5cc3a8e47d7d0316ecc0179d99c8a7be",
        "get_materialize_kernel": "8deeca3844d44e771c2d7f69b8a0dc6481c096f8b2c23fa94f5926f457d2369c",
        "get_general_kernel": "a8bbe6aa75c6dd8064f3b60cbdcc6d8ea8d6d72946b5afe3082fb157a1e1bf2f",
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
