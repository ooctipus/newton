# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Ask the CUDA driver whether a stream capture may be paused for a conditional graph node.

:func:`warp.capture_if` ends the capture to record its branch and resumes it
afterwards. Ending a capture fails while a stream forked into it holds captured
work the capturing stream has not joined (CUDA error 904), and Warp's own
``capture_end`` finds that work the same way as :func:`capture_has_unjoined_work`:
graph leaf nodes that are not among the stream's capture dependencies.
"""

import ctypes
import functools
import sys

_CUDA_SUCCESS = 0
_CAPTURE_STATUS_ACTIVE = 1


@functools.cache
def _driver():
    """Return the driver's capture and graph queries, or ``None`` when they cannot be loaded."""
    try:
        library = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
        capture_info = library.cuStreamGetCaptureInfo_v2
        graph_nodes = library.cuGraphGetNodes
        graph_edges = library.cuGraphGetEdges
    except (OSError, AttributeError):
        return None
    nodes = ctypes.POINTER(ctypes.c_void_p)
    count = ctypes.POINTER(ctypes.c_size_t)
    capture_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(nodes),
        count,
    ]
    graph_nodes.argtypes = [ctypes.c_void_p, nodes, count]
    graph_edges.argtypes = [ctypes.c_void_p, nodes, nodes, count]
    for query in (capture_info, graph_nodes, graph_edges):
        query.restype = ctypes.c_int
    return capture_info, graph_nodes, graph_edges


def capture_queries_available() -> bool:
    """Whether the CUDA driver's stream capture and graph queries can be called."""
    return _driver() is not None


def capture_has_unjoined_work(device) -> bool:
    """Whether the capture on ``device``'s current stream holds work the stream has not joined.

    Returns ``False`` for a device without CUDA streams, and ``True`` when the
    driver cannot answer, so that callers keep the capture running.

    Args:
        device: The Warp device whose current stream is capturing.
    """
    if not device.is_cuda:
        return False
    driver = _driver()
    if driver is None:
        return True
    capture_info, graph_nodes, graph_edges = driver
    status = ctypes.c_int()
    graph = ctypes.c_void_p()
    dependencies = ctypes.POINTER(ctypes.c_void_p)()
    count = ctypes.c_size_t()
    result = capture_info(
        ctypes.c_void_p(device.stream.cuda_stream),
        ctypes.byref(status),
        None,
        ctypes.byref(graph),
        ctypes.byref(dependencies),
        ctypes.byref(count),
    )
    if result != _CUDA_SUCCESS or status.value != _CAPTURE_STATUS_ACTIVE or not graph.value:
        return True
    joined = {dependencies[i] for i in range(count.value)}
    node_count = ctypes.c_size_t()
    if graph_nodes(graph, None, ctypes.byref(node_count)) != _CUDA_SUCCESS:
        return True
    nodes = (ctypes.c_void_p * node_count.value)()
    if node_count.value and graph_nodes(graph, nodes, ctypes.byref(node_count)) != _CUDA_SUCCESS:
        return True
    edge_count = ctypes.c_size_t()
    if graph_edges(graph, None, None, ctypes.byref(edge_count)) != _CUDA_SUCCESS:
        return True
    sources = (ctypes.c_void_p * edge_count.value)()
    targets = (ctypes.c_void_p * edge_count.value)()
    if edge_count.value and graph_edges(graph, sources, targets, ctypes.byref(edge_count)) != _CUDA_SUCCESS:
        return True
    parents = set(sources[: edge_count.value])
    return any(node not in parents and node not in joined for node in nodes[: node_count.value])
