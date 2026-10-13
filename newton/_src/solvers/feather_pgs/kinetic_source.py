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
    "allocation_count_source.py": {
        "_SCAN": "f323e929a00d5f65691b4240310a95665ce745a0bb9aa23b200609b537d2ff40",
        "_INITIALIZE": "00606fef8f4f5cba6bb6dfd7f8a90cea40c5c9585caa39f62aee61b4a977210e",
        "get_source": "db7f431a8fad785908909e573a860c04fc9350ad3c4caf6f612ea2b94e2119de",
        "bind": "a5ee7f4c2c21845d5368d9b8e8f088a6b458b22d5f1480fb180cedf54271fbde",
    },
    "contact_allocation_source.py": {
        "_BODIES": "b5476950d4d9abe81ea03201ffe8ac92ad46c5bc72e66a9fd85c104138e1db6b",
        "_WORLD": "f4d0d3bb212a2554312f3c6c100461fe6dd9f256d5a182c8b34014c9367020c1",
        "_PHI": "0075baa843e40e3c7c0ec4ddd26b14b397c47a73f011c2d414e403a275166c6a",
        "_RESOLVED": "54b6f3f3cd95fea186492a4144c1ce41aabebef2bf77e820afeefa580e6c95de",
        "_HELPER": "adb4b579ac2cb68406804ce53256dcccabf5aaa18ffa0942551b050096f08cf4",
        "_KERNEL": "55c3009578f93abf574ab1f76474eabc3e0cc2a7837b99eda5b31c486d216872",
        "get_source": "2a3f4729b68993d69e34e669d3cd8b0d417cdc8827baddd7ca46e8461b29d59f",
        "bind": "a9a3bc8bc1df49bbfb47c0a67c267a250bcac66a75210abf338cc0eb84252b1f",
    },
    "contact_geometry_source.py": {
        "_TRAVERSAL": "cb890e6ce2c0bd03bb710bc0a4bc5372cf563c4f9a8262daa0ba8a2d0fe4e53d",
        "_GEOMETRY": "fe0713e3b226fd7a47c6525c1f1648f838bb60ef6a9b05ff29cbdb4adb47a2af",
        "_check": "b4e5d25f276c6385a72478f47330d687ddc4f59987625ae53961f1011d773ff3",
        "get_geometry_source": "92c8ac5ce8c8aa4c6fcaf7d9cb1a35279f307eb9fb7350046ce291206b862cd1",
        "get_prelude": "9e29b02be9aa05dc5384587ac5f7807bc3766129c15fe5db1d66b68270fa40db",
        "get_stage_source": "7b073a856cb21f13120e53b25d10741215524a9391427c5baf86d01e3a7f682e",
    },
    "mf_dispatch.py": {
        "_NATIVE_ARGUMENTS": "795a55f6b5227fce9bba9d2df228dc0560ba344eb9e4da0227cb17684d59d28f",
        "_QUEUE_ARGUMENTS": "a3dce18c710b83d8747e3b4306183bcdf73a1cccf5c03913816ae1492dc9e947",
        "_compile": "ddcb24a52a1cf1d26c9609c818a5eaf69d480eff7374b640a1c0dbdddd697d43",
        "_signature": "902b3486f6499d0bad23cc6bb54c50aa6705ff9155c1f5493667e071c496ce81",
        "make_native": "a5d5100996cc1be1d8af2265552cfce787bdfa14c75409faf59f6c31324debfa",
        "make_dispatch": "7eb96a89eb255af78aa4e9e044f4b292d9957de5c49a8a390ee4a7dd6b9e3735",
    },
    "paired_factor_source.py": {
        "_KINETIC_ENTRY": "da56d0dccf7e1e8c073f7f8056919568b6acd71e7f47be9857b6bf577375ad4b",
        "_KINETIC_EXIT": "6a895fdf453c4b04cbaeb40e27208d660ac0d9db96afc4ef3b12fc35597a126a",
        "_KINETIC_ALIASES": "9441f2d4adb4aa4871e2f9e889c436064d60fdf3909dc6d1c10d9368c385c2df",
        "_physical_entry": "57b685e553d3b34f3a7958019b33e39d48308ac71faefeaddcda895f48f65bfb",
        "_physical_exit": "ee488b1935af525104f25aca4904ef3b26409ba9ec19baff61edef6d71966101",
        "get_source": "b4d0a4cb89f2fed72a72bb8934e3909ba31b0ed2241e621a2cf820a2d7f233eb",
    },
    "component_qualification.py": {
        "_PHYSICAL_PUBLISH": "d7129ba2edd6a8ebf30e104f99bc690be0ba8895de53eb4590a59cf3af235e87",
        "get_source": "5501780af8a2bb768c981245ce7ec0a110f8623bc5b2dd4ef44ebad4d7c3f2f2",
    },
    "small_cholesky.py": {
        "_FACTOR": "5d79f0480c1169dc55f13a27e2f78051dea24c8644a6de019f1e9866a78e8f8b",
        "_FORCE_INIT": "4e1883323ae63f673815d3f0e6aa1ef93365e126a1b6817f4cb2986e3c4faa3c",
        "_COMPOSITE_INPUT": "1ec974e47efd2095efa91bca708d53d96e78051dff51faddb44159efd9ba5b9d",
        "_INPUTS": "10b6aa85102d13efe7831ef5742108f1c689e04542dbb6bedcf9f3340e5f3ed4",
        "get_kernel": "9f15442b962d54b4a71384b9710bf3b497a78df1bdfb82fa1d0be4f6f24e468a",
    },
    "solver_feather_pgs.py": {
        "_get_pgs_solve_paired_factor_kernel": "e417ffcc1bd6dbae775e3fdd6943033bc27af1f46b1ec4ca28e1c7677d8d2bd6",
        "_get_pgs_solve_mf_gs_kernel": "35602b2592d4131f24f31518c17ccd5275684b81502311ee2faaabf1e10d9dd5",
        "_get_crba_cholesky_warp_kernel": "f507209b4a71155f75aa86a783241723e1370c41a0c262744c020bfc224969de",
        "_get_inverse_cholesky_register_kernel": "f1f2c60767d69d8006fb074c768877b921579954b7d36bc1efee9366212234d3",
        "_get_cholesky_kernel": "c262785fdd51c25f94c3c67ab4bb4cbe0fa95f16b61adcb788fbd16de10a4689",
        "_get_pack_mf_meta_kernel": "353265897c2af4cb85fc0e284b25bb9a862f67b54b74edd2e7b472a331c55a81",
    },
    "independent_components.py": {
        "get_prepare_kernel": "c60d23b924e8dbcdd21e039d9ede28c2f7bb3a415d961c36f25469608f8e62b6"
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
        "update_qdd_from_velocity": "d27c217ecec55796dea417dc5ec87ee576050e0a72ab2455aaca546ae293fb6c",
        "remove_free_root_transport_from_qdd": "ad0c81fd9713f999e434a13dd43bcb396f733c94cb655ef94f4f72c5023c6f21",
        "integrate_generalized_joints": "2418badf5019fb3b34eb3042e57835bda787a238c5227ff857ed950bed2d3a66",
        "jcalc_transform": "84d09399dec35112103e5b4bf315e81eddf5199a65075219d66a1f03df7e0996",
        "jcalc_motion": "15c927d5066b05e650fb66f1a28ccefbe8288218caba86a57d06da21e13bf608",
        "finalize_body_dynamics_values": "eb0862fab868f5d0bfcfcf3b78161fc4b96cc23a35744ecd8439e6e0fb9867ff",
        "finalize_body_dynamics_body": "16a2637f627c9b13948079cc96e8bc6d09fb7958f49f3f34c0a9a77672823d63",
    },
    "world_scan_publication.py": {
        "PublicationData": "fcf16576d1e626ccdb95ea13d55bbe84ec55a221d50705f1b03179e5f62d8452",
        "_POSE_SCAN": "6528a482804fa69088920e695a4d87f7f517bbb95cfce5c013f40041fc291317",
        "_MOTION_SCAN": "671291e49542ba1649414552ec76e7070d542842dfb2da21c93a74affbb9248c",
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
        "_TWISTS": "52394a8f739880c26884067834de9393e2b0f4f6da305b1ed38d4f5e4bf0c408",
        "_LIMITS": "1a402624e4030408efd39a7a1db85e03e08d91414381dc321c639cd9b62f7011",
        "_STORAGE": "99b0ce220a59a611ca8d8aa8ed6724d2d3a3bcc802e3408b6aa533600ceb9a89",
        "_limits": "ea79facce6db721b0beeaa40d1ced586fbac97a6fdd44ba92e548f1aa00f671a",
        "_ready": "dc63a432de2234472abe307c85ed98ad507381ad7baba67041c1fa082c2df6f0",
        "_storage": "ff666f78b27108b1285331fc44ba61b913e954aa0af9c0758525c30de9367db5",
        "_release": "58af3211560e7d4d26ca0b40e25d635e5ede62aec31568d600430a159fcb2715",
        "_lane": "07615c6d1dd5c43e7ad93a39b56a48e25b87498da0fb5c66403f5910c4b74745",
        "_body_twist": "5837c1ce0a3f402897e71aff09076f14880e2422111160df4264fa9ddef334be",
        "_body_valid": "be8cc8904be29e7a421895f91a51fa463612c244e00bd33a4481730683e7de71",
        "normal_source": "bc99413529d25a106471a8ee1250a4ed18a39be1b63f7697afcb52bd00ec38b6",
        "operation_source": "0ea35321bbd3b2f7ed8dac4c7a8edd04b7004887cf7f51aa55c23cd6f1bb0240",
        "_TWIST_PRODUCER": "e0b95d53727d39f634cf092c7b6d81c146a8be1a2c0cfc8b43efd283a78f06e3",
    },
    "simple_world.py": {
        "_SimpleWorldInput": "041b2334d5fd449d41a0d5f560b101942a4fd6932b8b38f9e07440a310c792ec",
        "_SimpleRawContacts": "93968df80ab45e14da99a8f1f5ab946b9214c9c2117f963102416491918f9c29",
        "_check_normal": "32a55038b911d18571ff35cec34b5bc37966d5ab5d74663c2d301732ec0cc2e9",
        "_endpoint_speed": "cf5eeed04958a2377b205b7bf4082729424911c47a9c359a435505f70a2c0b2a",
        "_finite3": "124a1b71c2f27d2e97d8261b2d8efda24591636e76f3607eb0d43b2ee934b227",
        "_finite6": "18d59e5041cf29be3fdf8bd1912bbf87bb7ac25227201ab347bd546dc7bfb6f7",
        "_passes": "45dffb35395ccfff2fefc32c3ec08245f40caacc0778e25aef7d588b81c7c8bd",
        "_normal_functions": "4b249d1a006c3510c2fd04d41c6c0fefd30c03446186de3cc0a4f5ff9d12616c",
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
        "_COMMON": "f46f089322434993a28cb07f076ad5c49a0698caba5797d706e6e3b8a6136fa3",
        "_CONTACT": "492e73fdab5f9875525c8f3e2752459c4b7d3b655e63ff6b26d5c7fbc27bc61d",
        "_t": "bac2d4805d420caba82de9e39ed1d5a3c33014f5c4fbbab9ce2d26bf6ade53ec",
        "_lane": "2612cc789289f0770732f9d03fb5575e8f29004133291745a61b5ab210c63636",
        "_release": "58af3211560e7d4d26ca0b40e25d635e5ede62aec31568d600430a159fcb2715",
    },
    "kinetic_state.py": {
        "_get_kernel": "b1804a99bdb0277de1c38e520b036970de88ded39f639c0bf8e21b2a6cdf0c01",
        "_world_gravity": "faffbf54ec341a09a0d99b459e6efe0956cc7b19026f6574c1bd3c99dc89806f",
        "_admit": "dad7435af0211e5b06822534a1003187a90f54ea997e4b20d5a52e123c85fe87",
        "_collect": "2c197cfa06bc5bbfeff5aa381ab6a7f594f61642a8b9bda765c4174e7dc1d6f6",
        "_COLLECT": "8409f401ce8985e5990e322896d36a57adee6185c5fb1f26c4f58ec7030e2b9b",
        "_primary_terms": "6b62b2c97692764bfc59feb026fdcd6482990a821a9882c9d409c2c4b0705374",
    },
    "kinetic_solve.py": {
        "_T": "03521fc92b11e045c58334667f9d55afbb9bd5b353ff4f5ecc0607def94db291",
        "_GUARD": "d05f161e7bd5d18f9a6a7de858fc6a1a1b77889b64498012a73bcc6ecf9d1cc0",
        "_CPU": "92408205c5c4295a291eb1ccd0b3b7a3a18538d0dac4b3f4949ee90aeff695d0",
        "_MATERIALIZE": "b76940135f6894f838e35be11ca2a88abc8365d1d610e49e5f29d3bb77eff2e3",
        "_impulse_transaction_edits": "5faa4439ad7e21d3e40bc06a6c7b54c59268ed278759264b898242e895b087b8",
        "recover_impulse_transactions": "3ba7137b8926c79662569883af2fae919af7dd9cf6fee9a6c2065e982021dd7a",
        "repair_impulse_transactions": "35fcc73dac25cabdd80259d4a76e2ecea2f08e7ec980822be8f18f8b893aed15",
        "cuda_recurrence": "ae9837209311b6f9e45b21723c65c3cbc568ce83e867c0c1acd9d9b6bf4a6fc8",
        "qualifier_source": "0e819d404f934e350e79e8fd22b854802ed14be007ab0e2b485354236d8c4822",
        "get_solve_kernel": "41401d6224f9e227e9509ff5f9b6343c1e7a3df33bf63b7590fecad5d0ca492e",
        "get_qualify_kernel": "e56490f1a74e258497378048176ee62dc658815ac0e5ad9a915cfad0ec3d04cb",
        "get_materialize_kernel": "60d020f8de5d16ebca4fc0fa1f22e073efd78ddda9df5247cb94d8b0f76b6717",
    },
    "kinetic_hybrid.py": {
        "HybridCoupledData": "1a261fa76dc5207521c424ac6489a29c87c9b6a0448add39538ea356af28a078",
        "_CPU": "c6f703863beb4837065c56bb4188a645a342e33c792308cefdb8eeb472168f84",
        "_STAGE": "d16346c926d401515e157795d0bd348e8f8114b635d6fe0b35b5e2739c83a4e0",
        "_STORE": "7abc04f9023ba89d1a529090ce8130f89a534ba0f577464bb0c5fc2c5c9c91ec",
        "qualify_source": "9ca4cbbd9b648ca81f1dd97cf5c7adcb35ca33de3d49c0f2958c396c66c35fcc",
        "general_source": "12dc88d4e7a753063ef1d279adbdecd1caaeeade0790c59f140fdf018eac519e",
        "get_qualify_kernel": "f524aab764befe22c841faf9b1b0e52a5cc3a8e47d7d0316ecc0179d99c8a7be",
        "get_materialize_kernel": "8deeca3844d44e771c2d7f69b8a0dc6481c096f8b2c23fa94f5926f457d2369c",
        "get_general_kernel": "6f30f0f656fd0484b4633f4a873612e9468c84ebe3d319c81882fd65d88b1ac2",
        "_fixed_source": "31b86b191c1c517ccbb36bb13cac5e2e8c097ed364c080d0ba74af511f2f69ec",
    },
    "cached_state.py": {
        "_INTEGRATION": "33f3c5af040aa39de98421346673484f257af0bd542d68b4098137e204c65457",
        "_RELATIVE": "7f1820f6cfaabfdf6ddaf55726577cc2246e01d41022daf1159f74ef1837a8f0",
        "_BODY_TERMS": "ef664e8ed9bf3772cecfcdcec9e58ebcc92c7a11cbd4d33c32b15d5356622ed4",
        "generalized_operations": "243b23e904bda78216977a582b6a448b02ff1bcc2f40e03a3c025ebb601691db",
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
    "cached_state_scan.py": {
        "_POSE_SCAN_TEMPLATE": "77022694a55763e5d496d9088c43646b138a3eb54091bdf62a54be9c6a81626a",
        "_MOTION_SCAN_TEMPLATE": "9827d2152eca7894e159c788f8f368550f9b394345f6a802223e92657eb730dc",
        "scan_source": "18e8b2533f9b3e5412d7e3c4d95e987a96eacb941fd0abebba0a68de687d580c",
    },
    "contact_normal.py": {
        "_ENDPOINT": "1b4fd9ec8f1d47ec32e4710d41fc73806e028aa762e8bb61cfae106fcb75092b",
        "_OWNERSHIP": "2dc5b3ceee2241a6c5429bd3ad45c8083b9dc93b0d01c611f0da9cf67819e37a",
        "_GEOMETRY": "b1ac0b20a3f5e4969b3fdbefca87aded766d7c2988e84c56192ecc750c516768",
        "_NORMAL": "b45153f88805c944304f40383aa4d9e4209a6944addb4b0baec3aff266c2fd14",
        "get_source": "ef918ff2e9ba669ff022a6effc43813a11c2dc5f9822ecd5cd08bc8591404323",
        "bind": "a7765296b5b9f1ec8a82e3fe724f7c3f3cb8e19ef85ae5eb57d327209f0d596e",
    },
    "body_validation.py": {
        "_PRESCRIBED": "d7663acb7de92486ba72d8a4d5f7a2c5301a99d3134f71407ee277db1cd10f9f",
        "_FREE": "8762eec1647003ef05ce924e1c27b7cd83970c3cb4f7faef08503b9867fe15f1",
        "_VALIDATION": "d21c270703aea141df92685d4d69ae9d9158ad34ed0a13d6ac5c5ae3f9f3e71c",
        "get_source": "14dce3fc0e8017e050e26483f302632279ed8ebeacf2cff0e7b767c1ca35e85f",
    },
    "mf_gs_source.py": {
        "GeneratedSource": "e7ab064517dd2cdfee617fb913870e0bb880abfc0836853b58ca3a5928f13447",
        "get_source": "4722e11f7da73a5e1cbfb7dbd237658085faccef2db9e399013cf17ba618c6a2",
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
