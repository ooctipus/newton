# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the mechanical port plus explicit transaction edits; no GPU claim."""

import ast
import hashlib
import importlib
import inspect
import textwrap
import unittest
from pathlib import Path

PACKAGE = "newton._src.solvers.feather_pgs."
SOURCE_PINS = {
    "kinetic_types": "cb4077d182a021399fb15c747501f111981fd06d72dc0f89ba2e96d0c0a2bf85",
    "kinetic_rows_types": "5446575e1f53cb411c037b2a52df9795eedc479c972906352dac30754245e34b",
    "kinetic_solve_types": "6ba20e1ee9e06115a930005de5a9ede932c6a87b52af1b790401378b854b8a43",
    "kinetic_state": "09d90363fa2c4157c215ac06c66bb4e305bd8df441fa57be1546564a58d4c3f9",
    "kinetic_predictor": "57b7b3bff720c248babd9148c835df1881f36e0f66c0c3ef09dc8e9d756adac6",
    "kinetic_rows": "6b13ed6d2be144a8e39fc84da4407b623dc798bb87334f2639d50d50340823e2",
    "kinetic_rows_triplet": "8bbf458d280d7447afc983b25669c5dc09c6fbaae4788981b8fd6ff03e234fcc",
    "kinetic_free_refresh": "2970588915ca4376cc2f6d4a553cfd255bd53f462b65e9c5ddb7da1ac4675626",
    "kinetic_services": "a5d3b8c1663a8aeb7a374425720ded46b281e338ae36f9068cd2371d0c2d18a8",
    "kinetic_allocate": "f68f36cd9f1e992fad66db0320be7304e8072a565ca5cfb3b5489caabd307380",
    "kinetic_zero": "209534c0799c2a6a2e4d5b2978c6228af79f10650bf6aad37932b52a726010c8",
    "kinetic_solve": "76077da44cb81fe09f36dbf04ed3bcbf395dd4d36fd72042dc08f8d88634b03d",
    "kinetic_guard": "34b1991513ab863eb090d8f6d26a70ce61acde1c411eaf19a5e1afc52046096f",
    "kinetic_public_force": "0fbb19d3a0d119e3998e9929c0ee1966def6022a1e5c2ebb5515eb2505123e44",
    "kinetic_hybrid": "0e7794cbbed06905f9cdc039e37c5ae9e30643b93b432783148548b3c9ba5265",
    "kinetic_compact_coupled": "84a6bd0479dd68c47b59381ef0cc23c0b4b921c9b6dfda7c05a4148c943b88d9",
}
EXPECTED = {
    "kinetic_allocate/AllocationData/descriptor": "c84add405b81f44f7fa533419523012d9adffe07da53c1a5edbc024d0eb53f04",
    "kinetic_allocate/initialize_count/python": "9f967b9f1bf4eb3f28a27d5c9945b793a8e11c0cf6683880e7b05c2b6c04d6a9",
    "kinetic_compact_coupled/_SIX_BRIDGE/literal": "f7669f8e6de34f78ec50765c7134b278f5f04c9df5556597d8d7ff567cb50de7",
    "kinetic_compact_coupled/_VIEWS/literal": "9749c77b5b82a8b912f27b1336b21bdf101149fed58a804860896c89c49b9ff2",
    "kinetic_compact_coupled/_arena/native": "b893e043cf040a4ba1d2d3b61f987a3be70e859ac590918a9716d919453a6081",
    "kinetic_compact_coupled/_arena/python": "6a4624196742eb065cfbe561ea98c3d2b5835f6210877c5a66fd1946ec2fc158",
    "kinetic_compact_coupled/get_general_kernel/closure/_arena": "b893e043cf040a4ba1d2d3b61f987a3be70e859ac590918a9716d919453a6081",
    "kinetic_compact_coupled/get_general_kernel/closure/_is_cpu": "599e3cf521417116b73f62d2c175b700d4631010feecc7a5658d7f0281c9a6ab",
    "kinetic_compact_coupled/get_general_kernel/closure/kinetic_compact_native": "16ff8f0465a3398dcc83130e43fb9e7a79cea84684b7e170c64e7e9496e5b170",
    "kinetic_compact_coupled/get_general_kernel/closure/kinetic_original_arena_native": "a9111e7f9865153aca6159f29f514bfc647e471024c1cd4ead84cdbb7f55bd42",
    "kinetic_compact_coupled/get_general_kernel/compact_mf_source": "3efe747c8f1c330151ce566d9f6b2a9747f4ed82585e2eb2ba94c5c920041af7",
    "kinetic_compact_coupled/get_general_kernel/compact_original_source": "1ea22a605c07900eb7365cf8c7a28b3dedc08de7ccefe3a0a7257c108882c649",
    "kinetic_compact_coupled/get_general_kernel/compact_source": "16ff8f0465a3398dcc83130e43fb9e7a79cea84684b7e170c64e7e9496e5b170",
    "kinetic_compact_coupled/get_general_kernel/python": "244c5095a5a6bb25a6932132fad8faf30685137c5593b86183acd01eed4c7bc9",
    "kinetic_compact_coupled/get_materialize_kernel/closure/materialize": "39b8c87e2a4c010c763a30267ac8027ee0f52db003eabd68d90a2264d8881f90",
    "kinetic_compact_coupled/get_materialize_kernel/python": "ef4f17b1a2ed3d0b2976c606e948623516d15d6a3d463aab6a4024daac559218",
    "kinetic_compact_coupled/get_qualify_kernel/closure/qualify": "7d97f5185d4ab899d9feb20d2b3803879a41cbcec01b7ccf9edd81d7894d1222",
    "kinetic_compact_coupled/get_qualify_kernel/python": "93f78eabfb407def2a88c019ee429010c62478d315a50217a65cb4fa6a080907",
    "kinetic_free_refresh/assemble_free_cpu_reference/python": "7bff7b3760bc21c52249101185672467d6c2a41e6da8a8412b4b7ee26f7232d7",
    "kinetic_free_refresh/inverse_lower_cpu_reference/python": "c96076f6099f6ddb64821853aeb44557134f4f37c86869e278aed6d4b52d0c4d",
    "kinetic_free_refresh/prepare_free_refresh/python": "dd08397d59b1f83d74cc907ceb1e9c695c077b09bdc4f53d6ca634c61a434960",
    "kinetic_guard/BoundaryGuard/descriptor": "865661616edcb8c5b531c677edfeb058e94d7bd9b697cfb44fb90a01cc516cfe",
    "kinetic_guard/_check/python": "3060baf7c3a71c36b3db21c4332d55a7d215cb0caa95f88b746bd4f4a57e7e39",
    "kinetic_guard/_guard_check/python": "0639010d3661ec14b5ef9051e55916e934b27f31da5dabef3e65a5244adeaba7",
    "kinetic_guard/_guard_world_check/closure/_uniform_error": "3bda475e656b941bc2bb652e3f910220e5c4aa2b97a5736695c7040fd63f8f78",
    "kinetic_guard/_guard_world_check/python": "b2ed25c5bc3dbccac2913a32fd2281c618ec6c584910ca252d36918fd3c2c06a",
    "kinetic_guard/_initialize/python": "cf07c7b1307245daf59e7c25ba23a1dea94a6990f2ec93b9ea2aef1493df3440",
    "kinetic_guard/_uniform_error/native": "3bda475e656b941bc2bb652e3f910220e5c4aa2b97a5736695c7040fd63f8f78",
    "kinetic_guard/_uniform_error/python": "fde71ce338afe816c19b76e4f134041c9dd9e2b5087ecbbf5bb7ca1a19799b6c",
    "kinetic_guard/get_finish_kernel/closure/_admit": "0ca89cfb89e6b3cf5ee556d6004726299d53bb6cf77dc958a6d71da9f39e02d9",
    "kinetic_guard/get_finish_kernel/closure/_collect": "dcccadd4d2fd79f663959f2a8aa5edcf1413adfb3fe852d9e829a299185fa3cf",
    "kinetic_guard/get_finish_kernel/closure/_lane": "16000b7469d2cf4a50a76c2c80f419eca131d4b7bb66f3ad0125cb54239eceb1",
    "kinetic_guard/get_finish_kernel/closure/_load_motion": "e52b55b11a7cc00cc26c44aa36e49ad5a6729b440083f265183ddf222dc6119d",
    "kinetic_guard/get_finish_kernel/closure/_load_origin": "333b498f162eebf26836251fd20ba45088a536160dd09f542b195d48390edf87",
    "kinetic_guard/get_finish_kernel/closure/_load_pose": "e7cf9de0f6ab2e5df6fb9e0a444e832b26699a47da0e64a7b7c132f074a6398e",
    "kinetic_guard/get_finish_kernel/closure/_load_shift": "8e5690e1e08f99590cf09b1f8d42f5689d4276a8a2e20ca33910b21ed9df4490",
    "kinetic_guard/get_finish_kernel/closure/_release": "96d0c7b488b5de3c7a420bb13b10d305441a12979033ae4fa36b78557e6ac20a",
    "kinetic_guard/get_finish_kernel/closure/_scan_motion": "783260be6c4203a6a6857b5741dec1abcecc2959fe965856d317e3feab63441e",
    "kinetic_guard/get_finish_kernel/closure/_scan_poses": "4b3f8080931a79adebd8c9b1bcb400d82dbec3a959e1173b5e109a0b5a594f5e",
    "kinetic_guard/get_finish_kernel/closure/_storage": "7b20bc9a902011331766078a401c654d37cd63f481f058cb9f02c4c774b87c53",
    "kinetic_guard/get_finish_kernel/closure/_store_axis": "d51b4845cc3e0a9b1781537feb2da9b3f397d629b323c10a222ff99a407044ba",
    "kinetic_guard/get_finish_kernel/closure/_store_motion": "39a34d8f6c33011732e09735ec22d5e8a306f2fb9c096f3caefc0ea1a20ed634",
    "kinetic_guard/get_finish_kernel/closure/_store_origin": "0df3115be684e7553a0b0057a94f843a9703968cbdae4144a5dde5c5a5930e6f",
    "kinetic_guard/get_finish_kernel/closure/_store_pose": "8328fca7e808a60d09cc4c229a62e1bb871b7368c62a43496958ebeccd13be18",
    "kinetic_guard/get_finish_kernel/closure/_store_shift": "f6b6d8e1c6185c617ec49038d88ad61c4100dffc6c5c24048b2f6a577a8afc3f",
    "kinetic_guard/get_finish_kernel/closure/_sync": "7d87c36cf4126ba2d3101bb2728465b36b1a84119b6d9be8e2980a936f4c94b8",
    "kinetic_guard/get_finish_kernel/python": "610fd7c6065fb879f9023cb855acdbb272541455c46de3af373b3c1ddb523f6a",
    "kinetic_guard/get_general_kernel/closure/pgs_solve_mf_gs_native": "1ea22a605c07900eb7365cf8c7a28b3dedc08de7ccefe3a0a7257c108882c649",
    "kinetic_guard/get_general_kernel/python": "6a49d242c3817c620a44cac77364c2921ba7299bc837b82cbe7fdc681544c28f",
    "kinetic_hybrid/HybridCoupledData/descriptor": "1a261fa76dc5207521c424ac6489a29c87c9b6a0448add39538ea356af28a078",
    "kinetic_hybrid/_CPU/literal": "e7e391d9a7653f4703a84d79da40cc7d4d42994990a9a0c1d821c601c5979316",
    "kinetic_hybrid/_STAGE/literal": "aff1df90537aaa5df4193cd57dd4dcb5ab286022d763bace028f1bd6999cbe58",
    "kinetic_hybrid/_STORE/literal": "e14f17abb62d9ff7a51ebfcf30c9a45944f640da23eda23fb03a87f024903d80",
    "kinetic_hybrid/get_general_kernel/closure/_is_cpu": "599e3cf521417116b73f62d2c175b700d4631010feecc7a5658d7f0281c9a6ab",
    "kinetic_hybrid/get_general_kernel/closure/kinetic_hybrid_native": "1c1c41d6dc2d95ed009f2b7a4f55946a91a075f6d58469d6ca95b17f2d343a10",
    "kinetic_hybrid/get_general_kernel/closure/pgs_solve_mf_gs_native": "1ea22a605c07900eb7365cf8c7a28b3dedc08de7ccefe3a0a7257c108882c649",
    "kinetic_hybrid/get_general_kernel/hybrid_original_source": "1ea22a605c07900eb7365cf8c7a28b3dedc08de7ccefe3a0a7257c108882c649",
    "kinetic_hybrid/get_general_kernel/hybrid_source": "1c1c41d6dc2d95ed009f2b7a4f55946a91a075f6d58469d6ca95b17f2d343a10",
    "kinetic_hybrid/get_general_kernel/python": "52c299afe9b78466e1fddd8e78958566340bc6ca06cf3c8fc56b1615fb93f341",
    "kinetic_hybrid/get_materialize_kernel/closure/materialize": "57cd055ea248b1c8739af80da633649e85b9f06ec7265d5ece0a758ed08939fa",
    "kinetic_hybrid/get_materialize_kernel/python": "986dc418dda29758beddfeb923c254aac30e07ca141394b704d6311aa7f915ff",
    "kinetic_hybrid/get_qualify_kernel/closure/qualify": "7d97f5185d4ab899d9feb20d2b3803879a41cbcec01b7ccf9edd81d7894d1222",
    "kinetic_hybrid/get_qualify_kernel/python": "93f78eabfb407def2a88c019ee429010c62478d315a50217a65cb4fa6a080907",
    "kinetic_predictor/_PREDICTOR/literal": "ffeb2009102fa3de150a2af22f0693b1353922e4b53553ec5a7998b7b7b458bc",
    "kinetic_predictor/_REFRESH/literal": "90552164c723a31cd61e8df44bb32fb7db1725730d6932d78a2e28e24bcaa2c3",
    "kinetic_predictor/_lane/native": "2db4bbd288242f40f695e7037bdd3cab38371ad0d68f70239daea714962341bc",
    "kinetic_predictor/_lane/python": "6761c643149a8fe1ea3f751f1422e46082d0e24e952a4c25c9bcb28121c60fe7",
    "kinetic_predictor/_predictor/native": "ffeb2009102fa3de150a2af22f0693b1353922e4b53553ec5a7998b7b7b458bc",
    "kinetic_predictor/_predictor/python": "fd95064cd0dbb8d3f7943a76b3a7be65a821d70b7aca6c1391cbccb1826e0490",
    "kinetic_predictor/_refresh/native": "90552164c723a31cd61e8df44bb32fb7db1725730d6932d78a2e28e24bcaa2c3",
    "kinetic_predictor/_refresh/python": "d8856d3fe09c21ceb796074c4ee4dc2656daa0000d5970877c2eef6797d44e3a",
    "kinetic_predictor/_release/native": "96d0c7b488b5de3c7a420bb13b10d305441a12979033ae4fa36b78557e6ac20a",
    "kinetic_predictor/_release/python": "1e871402e77c7bcedaecd29bacc30aa78b19c5d7d8ab2b6356074e5ef850f9c2",
    "kinetic_predictor/_storage/native": "16ba5f092581c9ab89e6150d7895056014f1c132a18ddea932c4eb457426000c",
    "kinetic_predictor/_storage/python": "eebb16155ea5699e4df622f0dbf3d5f8544921d837dd93939a10f31d01184a8f",
    "kinetic_predictor/get_predictor_kernel/closure/_lane": "2db4bbd288242f40f695e7037bdd3cab38371ad0d68f70239daea714962341bc",
    "kinetic_predictor/get_predictor_kernel/closure/_predictor": "ffeb2009102fa3de150a2af22f0693b1353922e4b53553ec5a7998b7b7b458bc",
    "kinetic_predictor/get_predictor_kernel/closure/_release": "96d0c7b488b5de3c7a420bb13b10d305441a12979033ae4fa36b78557e6ac20a",
    "kinetic_predictor/get_predictor_kernel/closure/_storage": "16ba5f092581c9ab89e6150d7895056014f1c132a18ddea932c4eb457426000c",
    "kinetic_predictor/get_predictor_kernel/python": "762d18e85a1654968985a7646d9fcf63521ece1178c250d7ba9b4c686913f0a4",
    "kinetic_predictor/get_refresh_kernel/closure/_lane": "2db4bbd288242f40f695e7037bdd3cab38371ad0d68f70239daea714962341bc",
    "kinetic_predictor/get_refresh_kernel/closure/_refresh": "90552164c723a31cd61e8df44bb32fb7db1725730d6932d78a2e28e24bcaa2c3",
    "kinetic_predictor/get_refresh_kernel/closure/_release": "96d0c7b488b5de3c7a420bb13b10d305441a12979033ae4fa36b78557e6ac20a",
    "kinetic_predictor/get_refresh_kernel/closure/_storage": "16ba5f092581c9ab89e6150d7895056014f1c132a18ddea932c4eb457426000c",
    "kinetic_predictor/get_refresh_kernel/python": "fedc9ba0d541f63bf0245337ed855fc305d452fc5b22a2d2ded0d6c73c0fd04c",
    "kinetic_public_force/compute_contact_linear_force_from_impulses/python": "5a3d9383ccab6da3d3b9c2c24ad1fb9da6b483d336db8d5f0a89740e9d23fcaa",
    "kinetic_public_force/pack_contact_linear_force_as_spatial/python": "1bfe0e1b52c1b2aa418bd03a95aebf5126e6e59bb6721cde0dc4643d5b3d1228",
    "kinetic_rows/_COMMON/literal": "73fab0b706a6b20478b3d572e837631f32da4b07f9d98329674712b4f5f2da19",
    "kinetic_rows/_CONTACT/literal": "56376571280ce53586a306aa016045a8cd079d44f8f22d486ca16a8de554e3ce",
    "kinetic_rows/_PREFIX/literal": "1a5d74343cffc408f40c9965fdc5fcd6934f4ad209e9b1e48b5647f65b1c6f97",
    "kinetic_rows/_contact/native": "56376571280ce53586a306aa016045a8cd079d44f8f22d486ca16a8de554e3ce",
    "kinetic_rows/_contact/python": "f86b7745dcc655a8f1040944abe9bc41777436d4038fcddf02162cac3c53ad09",
    "kinetic_rows/_lane/native": "16000b7469d2cf4a50a76c2c80f419eca131d4b7bb66f3ad0125cb54239eceb1",
    "kinetic_rows/_lane/python": "6761c643149a8fe1ea3f751f1422e46082d0e24e952a4c25c9bcb28121c60fe7",
    "kinetic_rows/_prefix/native": "1a5d74343cffc408f40c9965fdc5fcd6934f4ad209e9b1e48b5647f65b1c6f97",
    "kinetic_rows/_prefix/python": "e8215d7e80293faf2ea3908fe7339fe4ad9f4ed5fd53498e94154be9940955d8",
    "kinetic_rows/_release/native": "96d0c7b488b5de3c7a420bb13b10d305441a12979033ae4fa36b78557e6ac20a",
    "kinetic_rows/_release/python": "1e871402e77c7bcedaecd29bacc30aa78b19c5d7d8ab2b6356074e5ef850f9c2",
    "kinetic_rows/_storage/native": "a9ef6dc1115bf7be6dc4ba6d0c1dd00edca4c16991cd795a883f7fb63c44ac88",
    "kinetic_rows/_storage/python": "eebb16155ea5699e4df622f0dbf3d5f8544921d837dd93939a10f31d01184a8f",
    "kinetic_rows/_t/python": "0e1e4ecad5b00d3aebc6719ca213b66319d3667cb6e8ac4af124adc1681493c5",
    "kinetic_rows/get_arm_kernel/closure/_lane": "16000b7469d2cf4a50a76c2c80f419eca131d4b7bb66f3ad0125cb54239eceb1",
    "kinetic_rows/get_arm_kernel/python": "415536dae7c13921a89215f7f96970345fed8a35b05ed9d1657b919d95b10712",
    "kinetic_rows/get_contact_kernel/closure/_contact": "56376571280ce53586a306aa016045a8cd079d44f8f22d486ca16a8de554e3ce",
    "kinetic_rows/get_contact_kernel/closure/_lane": "16000b7469d2cf4a50a76c2c80f419eca131d4b7bb66f3ad0125cb54239eceb1",
    "kinetic_rows/get_contact_kernel/closure/_release": "96d0c7b488b5de3c7a420bb13b10d305441a12979033ae4fa36b78557e6ac20a",
    "kinetic_rows/get_contact_kernel/closure/_storage": "a9ef6dc1115bf7be6dc4ba6d0c1dd00edca4c16991cd795a883f7fb63c44ac88",
    "kinetic_rows/get_contact_kernel/python": "13989fac22b2f552fa9d14eb0b25fcee464821aebad8b0794dcb80b4bd11e213",
    "kinetic_rows/get_prefix_kernel/closure/_lane": "16000b7469d2cf4a50a76c2c80f419eca131d4b7bb66f3ad0125cb54239eceb1",
    "kinetic_rows/get_prefix_kernel/closure/_prefix": "1a5d74343cffc408f40c9965fdc5fcd6934f4ad209e9b1e48b5647f65b1c6f97",
    "kinetic_rows/get_prefix_kernel/closure/_release": "96d0c7b488b5de3c7a420bb13b10d305441a12979033ae4fa36b78557e6ac20a",
    "kinetic_rows/get_prefix_kernel/closure/_storage": "a9ef6dc1115bf7be6dc4ba6d0c1dd00edca4c16991cd795a883f7fb63c44ac88",
    "kinetic_rows/get_prefix_kernel/python": "2fe8d038b3dcd28c93504c74d5f8c31bbbb0110181ba577aa919d165d947f312",
    "kinetic_rows/get_validate_kernel/closure/_lane": "16000b7469d2cf4a50a76c2c80f419eca131d4b7bb66f3ad0125cb54239eceb1",
    "kinetic_rows/get_validate_kernel/python": "6c0d41e34fec52bb1ed0e41cfeb9b02f4d0664416c2a96d7520f81149c55b76d",
    "kinetic_rows_triplet/_CONTACT/literal": "426066e5b4cf77743aabe0afb39316088ed16719e77b231095ff1347c9694307",
    "kinetic_rows_triplet/_PRELUDE/literal": "62571683ea46ac856b0215e7521172bd3c73af459dac4cbb7eb37c15ff1def2e",
    "kinetic_rows_triplet/_TRIPLET/literal": "0eb2d882e631a68e1a665f7022b184271cdb4acbbe4acc1840970a7678f548a6",
    "kinetic_rows_triplet/_contact/native": "426066e5b4cf77743aabe0afb39316088ed16719e77b231095ff1347c9694307",
    "kinetic_rows_triplet/_contact/python": "f86b7745dcc655a8f1040944abe9bc41777436d4038fcddf02162cac3c53ad09",
    "kinetic_rows_triplet/_storage/native": "927086c1014aa5bd5224573ff508eb809cd405b4bce5e67f477303a06442ccd4",
    "kinetic_rows_triplet/_storage/python": "eebb16155ea5699e4df622f0dbf3d5f8544921d837dd93939a10f31d01184a8f",
    "kinetic_rows_triplet/get_contact_kernel/closure/_contact": "426066e5b4cf77743aabe0afb39316088ed16719e77b231095ff1347c9694307",
    "kinetic_rows_triplet/get_contact_kernel/closure/_storage": "927086c1014aa5bd5224573ff508eb809cd405b4bce5e67f477303a06442ccd4",
    "kinetic_rows_triplet/get_contact_kernel/python": "46a15d7ff688013b2900d3f9c5db6c228d65847ba1196b58b94d9bd2a8224b3a",
    "kinetic_rows_types/ArmMap/descriptor": "5578359c824125c2ca25daa48463553887dd0ae214071ee750e4b30efccb8051",
    "kinetic_rows_types/DenseRowOutput/descriptor": "c454d5769e5753fb643fb6d54b22eaa4566c91e3a618a0d34084e3f81dbbcd7e",
    "kinetic_rows_types/PrefixInput/descriptor": "181bb7f357a9c3481ef533c681d93a1f5961166122ce49b6d595abfc1a5acbaa",
    "kinetic_rows_types/RawRowInput/descriptor": "c4b7e9e416929a624afcf0b81573d8ac1e5766c822be23e532d5039ae5ecc9ff",
    "kinetic_rows_types/RowSettings/descriptor": "6c3ea0ef7aa1e6d0ca4056c7d758de00ad45df0f3b55c9ebc6adca7980adba83",
    "kinetic_rows_types/RowState/descriptor": "94ab81ec26595963443669013382ddef89fa736de8da7ff60c5965d64be9b325",
    "kinetic_services/_float_bits/native": "854a457d7e3955344693370f032abb1add448f7c52042c4e8ecf120510734c20",
    "kinetic_services/_float_bits/python": "bca3253f08e7efc0368987bff01b4286668b5ef8daaac53912facf036804d543",
    "kinetic_services/pack_mf_cpu_reference/closure/_float_bits": "854a457d7e3955344693370f032abb1add448f7c52042c4e8ecf120510734c20",
    "kinetic_services/pack_mf_cpu_reference/python": "e7076395f0b311afd78fe0bb330c154e9dac74e6a94170ca1eb8743480477a1f",
    "kinetic_solve/_CPU/literal": "d8f5071fb7e648b80a026a0936c3b43f65b59522bd0ea7bf4261603a6609a290",
    "kinetic_solve/_GUARD/literal": "b62ed92e42aa5b04767b9944501d354a09accde5192731dffb05553ddbb8d9a4",
    "kinetic_solve/_MATERIALIZE/literal": "0a3a38107753704f472302a27eb4b2a526cbd7bc2cbb735be06f3129b0c7ca74",
    "kinetic_solve/_T/literal": "db6d306dd00115c6a983f4bf90d94729d357100b28879129c38940d33a58abc3",
    "kinetic_solve/_is_cpu/native": "599e3cf521417116b73f62d2c175b700d4631010feecc7a5658d7f0281c9a6ab",
    "kinetic_solve/_is_cpu/python": "5261c67e02a8ac11a86c3867304d72e3ebc58a76c1b8c2bbd50a67cf8df75046",
    "kinetic_solve/get_materialize_kernel/closure/materialize": "0a3a38107753704f472302a27eb4b2a526cbd7bc2cbb735be06f3129b0c7ca74",
    "kinetic_solve/get_materialize_kernel/python": "760b3d3111d98cb6b3faa802844ffbd1b9827696a45aa30ed6ff1aa0bcd6d00a",
    "kinetic_solve/get_qualify_kernel/closure/qualify": "011a127fa8c23309e6834cb97de74810ef4269e4a2ff09f31bc2661167807804",
    "kinetic_solve/get_qualify_kernel/python": "bb9aa7500f70fb08a2e8d7010e5fe910bc7deef679f92f5dc34139917d1ce905",
    "kinetic_solve/get_solve_kernel/closure/_is_cpu": "599e3cf521417116b73f62d2c175b700d4631010feecc7a5658d7f0281c9a6ab",
    "kinetic_solve/get_solve_kernel/closure/solve_native": "0f62f5a52492f06c812cfe63c404efb1693a40bff2ea70cbb35b53edbca9fe09",
    "kinetic_solve/get_solve_kernel/python": "d3a3bc9bce45db1951a465d751185491581f59cc8a8f3df1815f74913684f28c",
    "kinetic_solve_types/KineticMFData/descriptor": "1a4ef04213dd238c35a6b19b480269ae605fdc10fe39d28ecea5909264d6c3c6",
    "kinetic_solve_types/KineticSolveData/descriptor": "a5cedb9dba99514998eddfb15a9035a95160852c40542531e6a95fae2641772b",
    "kinetic_state/_COLLECT/literal": "dcccadd4d2fd79f663959f2a8aa5edcf1413adfb3fe852d9e829a299185fa3cf",
    "kinetic_state/_SUM_CPU/literal": "b3cc2194ca00d83cfe22d995d2e2fc61ec2b4af7983a08a6e738152883b62f80",
    "kinetic_state/_SUM_CUDA/literal": "0c7b192422ff618ffa018fa114f7be19d29befda0bf83dc22ce06a258142c3e0",
    "kinetic_state/_admit/native": "0ca89cfb89e6b3cf5ee556d6004726299d53bb6cf77dc958a6d71da9f39e02d9",
    "kinetic_state/_admit/python": "e7867b62ea88a408140ae296f12aab14153bddac8f45c55391c0b9055b3597bc",
    "kinetic_state/_collect/native": "dcccadd4d2fd79f663959f2a8aa5edcf1413adfb3fe852d9e829a299185fa3cf",
    "kinetic_state/_collect/python": "b4f57f6c6e830adedc95f28da64493c315ea415cbd014784c36258ab357bff61",
    "kinetic_state/_inertia_action/python": "d34df2cac51218ce7fb2194dfd0d1011dab8a609e41be78a4c8e85f715f07da2",
    "kinetic_state/_lane/native": "16000b7469d2cf4a50a76c2c80f419eca131d4b7bb66f3ad0125cb54239eceb1",
    "kinetic_state/_lane/python": "6761c643149a8fe1ea3f751f1422e46082d0e24e952a4c25c9bcb28121c60fe7",
    "kinetic_state/_load_motion/native": "e52b55b11a7cc00cc26c44aa36e49ad5a6729b440083f265183ddf222dc6119d",
    "kinetic_state/_load_motion/python": "846cd41cac5f0757aa2f956dac66412a0aa2f178a800ae1e36a4dc086557cc1f",
    "kinetic_state/_load_origin/native": "333b498f162eebf26836251fd20ba45088a536160dd09f542b195d48390edf87",
    "kinetic_state/_load_origin/python": "e3e2b385298e14ed26cdb1631331c069a20192e220d66ad2e6d1073d84f4a027",
    "kinetic_state/_load_pose/native": "e7cf9de0f6ab2e5df6fb9e0a444e832b26699a47da0e64a7b7c132f074a6398e",
    "kinetic_state/_load_pose/python": "68cf4ce27009f1db27f2d768fc559e169ecfb7de04d6fc181387cb7e77ee8314",
    "kinetic_state/_load_shift/native": "8e5690e1e08f99590cf09b1f8d42f5689d4276a8a2e20ca33910b21ed9df4490",
    "kinetic_state/_load_shift/python": "e7903112a861f4f0f40d78eb67fb29f3c441b886e724f8d63dad5d0236f55d93",
    "kinetic_state/_primary_terms/closure/_store_terms": "846e524df1119eee55ba03c9cf34244cbeaac8be23456de4d39d0d818196091c",
    "kinetic_state/_primary_terms/python": "faa192eede7e701c65c2c20ed439432f22c3701a1df651b11a6f088c0afbc766",
    "kinetic_state/_release/native": "96d0c7b488b5de3c7a420bb13b10d305441a12979033ae4fa36b78557e6ac20a",
    "kinetic_state/_release/python": "1e871402e77c7bcedaecd29bacc30aa78b19c5d7d8ab2b6356074e5ef850f9c2",
    "kinetic_state/_scan_motion/native": "783260be6c4203a6a6857b5741dec1abcecc2959fe965856d317e3feab63441e",
    "kinetic_state/_scan_motion/python": "3b384b34e8b226bd3e1e0184d1156f630d947ebc81db450a59186ebb3230b544",
    "kinetic_state/_scan_poses/native": "4b3f8080931a79adebd8c9b1bcb400d82dbec3a959e1173b5e109a0b5a594f5e",
    "kinetic_state/_scan_poses/python": "4da6edf4c2058be3e494374f5ea89ce0e827f76d84b662f31dfc374f95c61348",
    "kinetic_state/_storage/native": "7b20bc9a902011331766078a401c654d37cd63f481f058cb9f02c4c774b87c53",
    "kinetic_state/_storage/python": "eebb16155ea5699e4df622f0dbf3d5f8544921d837dd93939a10f31d01184a8f",
    "kinetic_state/_store_axis/native": "d51b4845cc3e0a9b1781537feb2da9b3f397d629b323c10a222ff99a407044ba",
    "kinetic_state/_store_axis/python": "aa703d8ef27dced73515cdce616f10ba2c63b51479ca068b5dc56002ecbdeaf7",
    "kinetic_state/_store_motion/native": "39a34d8f6c33011732e09735ec22d5e8a306f2fb9c096f3caefc0ea1a20ed634",
    "kinetic_state/_store_motion/python": "967ce34e7dd8ef5789c105f62428061b556934dd6b17be08d402df2898672b15",
    "kinetic_state/_store_origin/native": "0df3115be684e7553a0b0057a94f843a9703968cbdae4144a5dde5c5a5930e6f",
    "kinetic_state/_store_origin/python": "15eab385207f706fd6371c2630a1df9fe605f25ddc17a3d5d9d4a1cd5f3b79a1",
    "kinetic_state/_store_pose/native": "8328fca7e808a60d09cc4c229a62e1bb871b7368c62a43496958ebeccd13be18",
    "kinetic_state/_store_pose/python": "2523b94676511702f83b31fe06856768a29875e58e3aace9fa8e1f96b964e840",
    "kinetic_state/_store_shift/native": "f6b6d8e1c6185c617ec49038d88ad61c4100dffc6c5c24048b2f6a577a8afc3f",
    "kinetic_state/_store_shift/python": "e36946b907c0853c2757d0a6ccef27cfd8aa1dd3de3aec80774dd113c9d6ef11",
    "kinetic_state/_store_terms/native": "846e524df1119eee55ba03c9cf34244cbeaac8be23456de4d39d0d818196091c",
    "kinetic_state/_store_terms/python": "f7af6d7913f7cddd29612bb5c8c1a57e43254651dd0e22c81401942c9d40616b",
    "kinetic_state/_sync/native": "7d87c36cf4126ba2d3101bb2728465b36b1a84119b6d9be8e2980a936f4c94b8",
    "kinetic_state/_sync/python": "85d31a11634f01744fe446ff135723b0b9d39d4c194320220cd4c32871d5ff9b",
    "kinetic_state/get_construct_kernel/closure/_admit": "0ca89cfb89e6b3cf5ee556d6004726299d53bb6cf77dc958a6d71da9f39e02d9",
    "kinetic_state/get_construct_kernel/closure/_collect": "dcccadd4d2fd79f663959f2a8aa5edcf1413adfb3fe852d9e829a299185fa3cf",
    "kinetic_state/get_construct_kernel/closure/_lane": "16000b7469d2cf4a50a76c2c80f419eca131d4b7bb66f3ad0125cb54239eceb1",
    "kinetic_state/get_construct_kernel/closure/_load_motion": "e52b55b11a7cc00cc26c44aa36e49ad5a6729b440083f265183ddf222dc6119d",
    "kinetic_state/get_construct_kernel/closure/_load_origin": "333b498f162eebf26836251fd20ba45088a536160dd09f542b195d48390edf87",
    "kinetic_state/get_construct_kernel/closure/_load_pose": "e7cf9de0f6ab2e5df6fb9e0a444e832b26699a47da0e64a7b7c132f074a6398e",
    "kinetic_state/get_construct_kernel/closure/_load_shift": "8e5690e1e08f99590cf09b1f8d42f5689d4276a8a2e20ca33910b21ed9df4490",
    "kinetic_state/get_construct_kernel/closure/_release": "96d0c7b488b5de3c7a420bb13b10d305441a12979033ae4fa36b78557e6ac20a",
    "kinetic_state/get_construct_kernel/closure/_scan_motion": "783260be6c4203a6a6857b5741dec1abcecc2959fe965856d317e3feab63441e",
    "kinetic_state/get_construct_kernel/closure/_scan_poses": "4b3f8080931a79adebd8c9b1bcb400d82dbec3a959e1173b5e109a0b5a594f5e",
    "kinetic_state/get_construct_kernel/closure/_storage": "7b20bc9a902011331766078a401c654d37cd63f481f058cb9f02c4c774b87c53",
    "kinetic_state/get_construct_kernel/closure/_store_axis": "d51b4845cc3e0a9b1781537feb2da9b3f397d629b323c10a222ff99a407044ba",
    "kinetic_state/get_construct_kernel/closure/_store_motion": "39a34d8f6c33011732e09735ec22d5e8a306f2fb9c096f3caefc0ea1a20ed634",
    "kinetic_state/get_construct_kernel/closure/_store_origin": "0df3115be684e7553a0b0057a94f843a9703968cbdae4144a5dde5c5a5930e6f",
    "kinetic_state/get_construct_kernel/closure/_store_pose": "8328fca7e808a60d09cc4c229a62e1bb871b7368c62a43496958ebeccd13be18",
    "kinetic_state/get_construct_kernel/closure/_store_shift": "f6b6d8e1c6185c617ec49038d88ad61c4100dffc6c5c24048b2f6a577a8afc3f",
    "kinetic_state/get_construct_kernel/closure/_sync": "7d87c36cf4126ba2d3101bb2728465b36b1a84119b6d9be8e2980a936f4c94b8",
    "kinetic_state/get_construct_kernel/python": "a91d13c0fb6ef8ff5d369793690f6e9b14539a77338fdbab935187bf4d09ca36",
    "kinetic_state/get_finish_kernel/closure/_admit": "0ca89cfb89e6b3cf5ee556d6004726299d53bb6cf77dc958a6d71da9f39e02d9",
    "kinetic_state/get_finish_kernel/closure/_collect": "dcccadd4d2fd79f663959f2a8aa5edcf1413adfb3fe852d9e829a299185fa3cf",
    "kinetic_state/get_finish_kernel/closure/_lane": "16000b7469d2cf4a50a76c2c80f419eca131d4b7bb66f3ad0125cb54239eceb1",
    "kinetic_state/get_finish_kernel/closure/_load_motion": "e52b55b11a7cc00cc26c44aa36e49ad5a6729b440083f265183ddf222dc6119d",
    "kinetic_state/get_finish_kernel/closure/_load_origin": "333b498f162eebf26836251fd20ba45088a536160dd09f542b195d48390edf87",
    "kinetic_state/get_finish_kernel/closure/_load_pose": "e7cf9de0f6ab2e5df6fb9e0a444e832b26699a47da0e64a7b7c132f074a6398e",
    "kinetic_state/get_finish_kernel/closure/_load_shift": "8e5690e1e08f99590cf09b1f8d42f5689d4276a8a2e20ca33910b21ed9df4490",
    "kinetic_state/get_finish_kernel/closure/_release": "96d0c7b488b5de3c7a420bb13b10d305441a12979033ae4fa36b78557e6ac20a",
    "kinetic_state/get_finish_kernel/closure/_scan_motion": "783260be6c4203a6a6857b5741dec1abcecc2959fe965856d317e3feab63441e",
    "kinetic_state/get_finish_kernel/closure/_scan_poses": "4b3f8080931a79adebd8c9b1bcb400d82dbec3a959e1173b5e109a0b5a594f5e",
    "kinetic_state/get_finish_kernel/closure/_storage": "7b20bc9a902011331766078a401c654d37cd63f481f058cb9f02c4c774b87c53",
    "kinetic_state/get_finish_kernel/closure/_store_axis": "d51b4845cc3e0a9b1781537feb2da9b3f397d629b323c10a222ff99a407044ba",
    "kinetic_state/get_finish_kernel/closure/_store_motion": "39a34d8f6c33011732e09735ec22d5e8a306f2fb9c096f3caefc0ea1a20ed634",
    "kinetic_state/get_finish_kernel/closure/_store_origin": "0df3115be684e7553a0b0057a94f843a9703968cbdae4144a5dde5c5a5930e6f",
    "kinetic_state/get_finish_kernel/closure/_store_pose": "8328fca7e808a60d09cc4c229a62e1bb871b7368c62a43496958ebeccd13be18",
    "kinetic_state/get_finish_kernel/closure/_store_shift": "f6b6d8e1c6185c617ec49038d88ad61c4100dffc6c5c24048b2f6a577a8afc3f",
    "kinetic_state/get_finish_kernel/closure/_sync": "7d87c36cf4126ba2d3101bb2728465b36b1a84119b6d9be8e2980a936f4c94b8",
    "kinetic_state/get_finish_kernel/python": "a91d13c0fb6ef8ff5d369793690f6e9b14539a77338fdbab935187bf4d09ca36",
    "kinetic_types/CurrentForceInput/descriptor": "3e11c6dead9c5e53bdf3b35837e84c8b8d96299b4445cc7f5f2724a5f04f2f2a",
    "kinetic_types/CurrentKineticCache/descriptor": "d9a75025ec69e397f1568e3b9ef47fdc5cdaee83779548a9b56df7fa7fcfb85a",
    "kinetic_types/GeometricCache/descriptor": "facda8917ee868e6715299b00884323303c19722c3ffa36f69ec461712a80c12",
    "kinetic_types/HeldKineticOperator/descriptor": "77419e94778515a68824faaee4b9ca59cb15f1e8d1addc233d25845d9b027ed1",
    "kinetic_types/KineticPlan/descriptor": "0d50fb7c6811a407f0be26b5853612d6dfb0414d6c6752a74471c8d2440eb708",
    "kinetic_types/KineticPredictorOutput/descriptor": "ebe8470674e0e0acf12c13af49bcc8cb3e2eb395624aecee5f94923dcfb5b30d",
    "kinetic_types/KineticSchedule/descriptor": "d70bf1f94e82cb2b44f59b83e9dc653faec0843ac614adee3f7c9f56b3759593",
    "kinetic_types/RefreshInput/descriptor": "cb7bd054dbf9d4d935aae41e867aa4c403027846d685511bad6639a28ac4cb5f",
    "kinetic_zero/_PUBLISH/literal": "5af3acfd523a1d30dbf24064c3639ef731bf0d45c34d7cc23c3fc976c5a0e140",
    "kinetic_zero/_load/native": "99c23ee48e1ab48f71af20f42142b8bd585c6254156dbb162e39ac85d3abb609",
    "kinetic_zero/_load/python": "cee937a49f86340aa37e66a6fdbc5d47f85da46c6fe838f49bcefb6a0b33d400",
    "kinetic_zero/_publish/native": "5af3acfd523a1d30dbf24064c3639ef731bf0d45c34d7cc23c3fc976c5a0e140",
    "kinetic_zero/_publish/python": "a416c9e9a7f7d57d7f031fc596eb4e1a97e28ec92a699dc5f3299999a68155fb",
    "kinetic_zero/get_kernel/closure/_load": "99c23ee48e1ab48f71af20f42142b8bd585c6254156dbb162e39ac85d3abb609",
    "kinetic_zero/get_kernel/closure/_publish": "5af3acfd523a1d30dbf24064c3639ef731bf0d45c34d7cc23c3fc976c5a0e140",
    "kinetic_zero/get_kernel/python": "f6431fc39447f39af55ee85709111f73d01de1cd0e9072d410fc8c53bddd5104",
}
MODULES = (*SOURCE_PINS, "kinetic_source")
# Separately proven eleven-edit transaction correction. The original 244
# expectations above remain unchanged after exact reversal of only these edits.
TRANSACTION_CORRECTIONS = {
    "kinetic_solve/get_solve_kernel/closure/solve_native": "cbfe46962815812393fcb6e11c3c09b72c70037223ed71b31bdfc1821891e615",
    "kinetic_compact_coupled/get_general_kernel/compact_source": "a8a17dd80e7777c998ef027f270d5158a17873b6bd7d924c452c3f057868bb02",
    "kinetic_compact_coupled/get_general_kernel/closure/kinetic_compact_native": "a8a17dd80e7777c998ef027f270d5158a17873b6bd7d924c452c3f057868bb02",
}

# One register-only current-world gravity view precedes both original force
# consumers. Keep the original 244 records after reversing only that assignment.
WORLD_GRAVITY_PYTHON = {
    "kinetic_state/get_construct_kernel": "c0194c61d8bfbfc015217e0079b8fa51d8abd75473d14d159d5b76b70e917a5e",
    "kinetic_state/get_finish_kernel": "c0194c61d8bfbfc015217e0079b8fa51d8abd75473d14d159d5b76b70e917a5e",
    "kinetic_guard/get_finish_kernel": "75c7c23bc9c74825d29f69952b5eafc5e45713b0d295d4ae0c72d5858a70ff3e",
}
WORLD_GRAVITY_ADDED = {
    "kinetic_state/_world_gravity/native": "38cc4d45e3852c02ff7ebfd8015cd043bd6b0d072c1a518dba02e97abc3a5fa0",
    "kinetic_state/_world_gravity/python": "1df1f0457ab87c6570a2d6f0e9baaa1055344f440753d210c4af46d90e783acb",
    "kinetic_state/get_construct_kernel/closure/_world_gravity": "38cc4d45e3852c02ff7ebfd8015cd043bd6b0d072c1a518dba02e97abc3a5fa0",
    "kinetic_state/get_finish_kernel/closure/_world_gravity": "38cc4d45e3852c02ff7ebfd8015cd043bd6b0d072c1a518dba02e97abc3a5fa0",
    "kinetic_guard/get_finish_kernel/closure/_world_gravity": "38cc4d45e3852c02ff7ebfd8015cd043bd6b0d072c1a518dba02e97abc3a5fa0",
}


def source_oracle():
    """Collect exact native strings, generated dispatch ASTs and descriptor ABI."""
    modules = {name: importlib.import_module(PACKAGE + name) for name in SOURCE_PINS}
    source = importlib.import_module(PACKAGE + "kinetic_source")
    result = {}

    def digest(value):
        return hashlib.sha256(value.encode()).hexdigest()

    def native_digest(label, value):
        if label in TRANSACTION_CORRECTIONS:
            if digest(value) != TRANSACTION_CORRECTIONS[label]:
                raise ValueError("Unexpected transaction source changes: " + label)
            value = modules["kinetic_solve"].recover_impulse_transactions(value)
        elif any(marker in value for marker in ("IMPULSE_READ_COMPLETE", "SIBLING_READ_COMPLETE", "UNUSED_OWN_STORE")):
            raise ValueError("Transaction correction outside its declared owner: " + label)
        return digest(value)

    def inspect_owner(label, obj):
        for field in (
            "compact_source",
            "compact_original_source",
            "compact_mf_source",
            "hybrid_source",
            "hybrid_original_source",
        ):
            value = getattr(obj, field, None)
            if value is not None:
                result[label + "/" + field] = native_digest(label + "/" + field, value)
        native = getattr(obj, "native_snippet", None)
        if native is not None:
            result[label + "/native"] = native_digest(label + "/native", native)
        function = getattr(obj, "func", None)
        if function is not None:
            tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    node.decorator_list = []
            if label in WORLD_GRAVITY_PYTHON:
                if source.definition_digest(tree) != WORLD_GRAVITY_PYTHON[label]:
                    raise ValueError("Unexpected gravity-view source change: " + label)
                function_node = tree.body[0]
                assignments = [
                    node
                    for node in function_node.body
                    if isinstance(node, ast.Assign)
                    and ast.unparse(node) == "data.gravity = _world_gravity(data.gravity, world)"
                ]
                if len(assignments) != 1:
                    raise ValueError("Ambiguous current-world gravity assignment")
                function_node.body.remove(assignments[0])
            result[label + "/python"] = source.definition_digest(tree)
            closure = inspect.getclosurevars(function)
            for name, value in (closure.globals | closure.nonlocals).items():
                native = getattr(value, "native_snippet", None)
                if native is not None:
                    result[label + "/closure/" + name] = native_digest(label + "/closure/" + name, native)

    for name, module in modules.items():
        for node in ast.parse(Path(module.__file__).read_text()).body:
            if isinstance(node, ast.ClassDef) and any(
                ast.unparse(decorator) == "wp.struct" for decorator in node.decorator_list
            ):
                result[name + "/" + node.name + "/descriptor"] = source.definition_digest(node)
        for key, value in module.__dict__.items():
            if key.startswith("__"):
                continue
            if isinstance(value, str) and len(value) > 80 and (";" in value or "#if" in value):
                result[name + "/" + key + "/literal"] = digest(value)
            if hasattr(value, "func") and not inspect.ismodule(value):
                inspect_owner(name + "/" + key, value)

    factories = {
        "kinetic_state": ("get_construct_kernel", "get_finish_kernel"),
        "kinetic_predictor": ("get_predictor_kernel", "get_refresh_kernel"),
        "kinetic_rows": ("get_arm_kernel", "get_prefix_kernel", "get_contact_kernel", "get_validate_kernel"),
        "kinetic_rows_triplet": ("get_contact_kernel",),
        "kinetic_zero": ("get_kernel",),
        "kinetic_solve": ("get_solve_kernel", "get_qualify_kernel", "get_materialize_kernel"),
        "kinetic_guard": ("get_finish_kernel",),
        "kinetic_hybrid": ("get_qualify_kernel", "get_materialize_kernel"),
        "kinetic_compact_coupled": ("get_qualify_kernel", "get_materialize_kernel"),
    }
    for module, names in factories.items():
        for name in names:
            inspect_owner(module + "/" + name, getattr(modules[module], name)("cpu"))
    force = modules["kinetic_public_force"]
    for name in force.NAMES:
        inspect_owner("kinetic_public_force/" + name, force.get_kernel(name))
    solver = importlib.import_module(PACKAGE + "solver_feather_pgs")
    original = solver._get_pgs_solve_mf_gs_kernel(
        192,
        64,
        29,
        "cpu",
        has_drive_rows=False,
        has_dense_velocity_limit_rows=False,
        factor_coordinates=True,
        independent_components=True,
    )
    for module in ("kinetic_guard", "kinetic_hybrid", "kinetic_compact_coupled"):
        inspect_owner(module + "/get_general_kernel", modules[module].get_general_kernel(original))
    return result


class TestKineticNativePort(unittest.TestCase):
    """Keep module/source closure separate from live numerical acceptance."""

    def test_import_native_module_family(self):
        """Import all owners without an external scratch module search path."""
        for name in MODULES:
            with self.subTest(module=name):
                self.assertIsNotNone(importlib.import_module(PACKAGE + name))

    def test_frozen_native_and_descriptor_oracle(self):
        """Undo only explicit transaction edits, then recover every frozen record."""
        actual = source_oracle()
        expectations = EXPECTED | WORLD_GRAVITY_ADDED
        self.assertEqual(set(actual), set(expectations))
        for name, expected in expectations.items():
            with self.subTest(source=name):
                self.assertEqual(actual[name], expected)

    def test_impulse_transaction_correction(self):
        """Require the separately proven correction and byte-exact undo."""
        module = importlib.import_module(PACKAGE + "kinetic_solve")
        # Exact edit log extracted independently from the paired-GPU-passed
        # transaction adapter 1940286e7c86effb, not regenerated from this helper.
        self.assertEqual(
            hashlib.sha256(repr(module._impulse_transaction_edits()).encode()).hexdigest(),
            "fdcabd747dd84c9549e0d344d11b95e1fe8909df4ade79ba457bfa515e654913",
        )
        corrected = module.cuda_recurrence()
        original = module.recover_impulse_transactions(corrected)
        self.assertEqual(module.repair_impulse_transactions(original), corrected)
        self.assertEqual(corrected.count("SIBLING_READ_COMPLETE"), 3)
        self.assertEqual(corrected.count("IMPULSE_READ_COMPLETE_"), 5)
        self.assertEqual(corrected.count("UNUSED_OWN_STORE_"), 3)
        with self.assertRaises(ValueError):
            module.repair_impulse_transactions(corrected)
        with self.assertRaises(ValueError):
            module.recover_impulse_transactions(corrected.replace("IMPULSE_READ_COMPLETE_0", "changed"))

    def test_composed_default_preserves_original_native_bytes(self):
        """Keep the qualified Kuka ABI/body while retaining the optional branch."""
        solver = importlib.import_module(PACKAGE + "solver_feather_pgs")
        original = solver._get_pgs_solve_mf_gs_kernel(
            192,
            64,
            29,
            "cpu",
            has_drive_rows=False,
            has_dense_velocity_limit_rows=False,
            factor_coordinates=True,
            independent_components=True,
        )
        native = inspect.getclosurevars(original.func).nonlocals["pgs_solve_mf_gs_native"]
        self.assertEqual(
            hashlib.sha256(native.native_snippet.encode()).hexdigest(),
            "1ea22a605c07900eb7365cf8c7a28b3dedc08de7ccefe3a0a7257c108882c649",
        )
        self.assertEqual(original.key, "pgs_solve_mf_gs_192_64_29_current_drive0_densevlim0_factor_components")
        optional = solver._get_pgs_solve_mf_gs_kernel(
            100,
            64,
            43,
            "cpu",
            has_drive_rows=False,
            has_dense_velocity_limit_rows=False,
            single_factor_coordinates=True,
        )
        optional_native = inspect.getclosurevars(optional.func).nonlocals["pgs_solve_mf_gs_native"]
        self.assertEqual(inspect.signature(original.func), inspect.signature(optional.func))
        self.assertIn("_single_factor", optional.key)
        self.assertIn("s_rhs_dense[i] += initial_jv", optional_native.native_snippet)
        self.assertIn("s_v[d] = 0.0f", optional_native.native_snippet)
        # Independently recorded from the accepted baf0 G1 single-factor body.
        nonblank = "\n".join(line for line in optional_native.native_snippet.splitlines() if line.strip())
        self.assertEqual(
            hashlib.sha256(nonblank.encode()).hexdigest(),
            "f6fd09a5b4efde69f992196aa88d0ab1f84fbbd6d015c2a4443486f72f104e34",
        )

    def test_runtime_has_no_scratch_or_fixture_imports(self):
        """Reject external paths and capture-oracle imports in the runtime port."""
        for name in MODULES:
            module = importlib.import_module(PACKAGE + name)
            tree = ast.parse(Path(module.__file__).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertNotIn("/tmp/", node.value)
                    self.assertNotIn("/home/", node.value)
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(alias.name.startswith(("kinetic_", "test_")))
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("test_"))
                    self.assertNotIn("reference", node.module or "")
                    self.assertNotIn("fixture", node.module or "")
                    if (node.module or "").startswith("kinetic_"):
                        self.assertGreater(node.level, 0)

    def test_source_guard_allows_unrelated_solver_edits(self):
        """Allow unrelated class edits while rejecting retained math changes."""
        helper = importlib.import_module(PACKAGE + "kinetic_source")
        code = "def retained(x):\n    return x + 1\n\nclass Solver:\n    unrelated = 1\n"
        pin = helper.definition_digest(ast.parse(code).body[0])
        self.assertIn("retained", helper.check_source(code, {"retained": pin}))
        helper.check_source(code.replace("unrelated = 1", "unrelated = 2"), {"retained": pin})
        with self.assertRaises(RuntimeError):
            helper.check_source(code.replace("return x + 1", "return x + 2"), {"retained": pin})
        with self.assertRaises(ValueError):
            helper.checked_definitions("../solver_feather_pgs.py", ("retained",))

    def test_source_guard_definitions_are_private_copies(self):
        """Prevent an adapter from mutating the cached source authority."""
        helper = importlib.import_module(PACKAGE + "kinetic_source")
        filename, name = "kinetic_rows.py", "_CONTACT"
        first = helper.checked_definitions(filename, (name,))
        first[name].targets[0].id = "poison"
        second = helper.checked_definitions(filename, (name,))
        self.assertEqual(second[name].targets[0].id, name)


if __name__ == "__main__":
    unittest.main()
