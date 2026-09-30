# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepare exact CUDA kernel-node bindings for GPU-controlled Warp launch extents.

The caller proves the semantic world axis and scalar arguments. This owner handles
CUDA argument layout and graph resources, never infers semantics from numeric
values or kernel names. Preparation is host-side; replay updates run on the GPU.
"""

import ctypes as ct
import hashlib
import os
import shutil
import subprocess
import weakref
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp


@dataclass(frozen=True)
class Int32Parameter:
    index: int  # CUDA indices include the launch-bounds argument at zero.
    source: wp.array
    maximum: int


@dataclass(frozen=True)
class KernelBinding:
    node: int
    rank: int
    extent_axis: int | None = 0
    extent_source: wp.array | None = None
    parameters: tuple[Int32Parameter, ...] = ()


def _check(status, operation):
    if status:
        raise RuntimeError(f"{operation} failed with CUDA/binding status {status}")


def capture_parallel(branches, *, stream=None):
    """Capture independent callbacks as sibling paths, joining before subsequent work.

    This changes the dependency frontier of one real Warp-managed CUDA capture;
    it creates neither streams nor executable graphs. Callbacks must restore the
    same parent capture and use independent writable memory. Read-only inputs may
    be shared. Their resources must remain owned by the final graph. Nested Warp
    conditionals and nested calls are supported; APIC serialization is not.

    All CUDA queries/edits happen during preparation. After any callback or CUDA
    failure the caller must discard the capture, even if it catches the exception.
    DeviceGraphUpdates refuses to prepare or instantiate a failed capture.
    """
    branches = tuple(branches)
    if any(not callable(branch) for branch in branches):
        raise TypeError("Parallel capture branches must be callables")
    stream = wp.get_stream() if stream is None else stream
    device = stream.device
    owner = device.captures.get(stream) if device.is_cuda else None
    if owner is None or owner.device != device:
        raise RuntimeError("Parallel branches require a Warp-managed CUDA graph capture")
    if getattr(owner, "_parallel_capture_failed", False):
        raise RuntimeError("Discard this failed parallel capture")
    if getattr(owner, "apic", False):
        raise ValueError("Parallel dependency editing is not supported by APIC serialization")
    library = ct.CDLL("libcuda.so.1")
    pointer = ct.c_void_p
    library.cuStreamGetCaptureInfo_v2.argtypes = [
        pointer,
        ct.POINTER(ct.c_int),
        ct.POINTER(ct.c_uint64),
        ct.POINTER(pointer),
        ct.POINTER(ct.POINTER(pointer)),
        ct.POINTER(ct.c_size_t),
    ]
    library.cuStreamUpdateCaptureDependencies.argtypes = [pointer, ct.POINTER(pointer), ct.c_size_t, ct.c_uint]

    def frontier():
        if getattr(owner, "_parallel_capture_failed", False):
            raise RuntimeError("Discard this failed parallel capture")
        if device.captures.get(stream) is not owner:
            raise RuntimeError("Parallel callback changed the Warp capture owner")
        status, identity, graph = ct.c_int(), ct.c_uint64(), pointer()
        nodes, count = ct.POINTER(pointer)(), ct.c_size_t()
        _check(
            library.cuStreamGetCaptureInfo_v2(
                stream.cuda_stream,
                ct.byref(status),
                ct.byref(identity),
                ct.byref(graph),
                ct.byref(nodes),
                ct.byref(count),
            ),
            "query parallel capture frontier",
        )
        if status.value != 1 or not graph.value or identity.value != owner.capture_id:
            raise RuntimeError("Warp and native CUDA capture ownership disagree")
        return graph.value, tuple(nodes[i] for i in range(count.value))

    def replace(nodes):
        # CUDA copies these node handles; no pointer outlives this call.
        _check(
            library.cuStreamUpdateCaptureDependencies(
                stream.cuda_stream, (pointer * len(nodes))(*nodes), len(nodes), 1
            ),
            "set parallel capture frontier",
        )

    try:
        parent, prefix = frontier()
        joined = set()
        for branch in branches:
            replace(prefix)
            branch()
            current, tails = frontier()
            if current != parent:
                raise RuntimeError("Parallel callback did not restore its parent CUDA graph")
            # Warp resumes a conditional's parent with all parent leaf nodes;
            # retaining those older leaves in the union is a valid explicit join.
            joined.update(tails)
        replace(tuple(sorted(joined)) if branches else prefix)
    except BaseException:
        owner._parallel_capture_failed = True
        raise


class DeviceGraphUpdates:
    """Own one prepared descriptor table and its count input; bind once to one graph.

    Enable counts, launch extents and scalar parameters have explicit independent
    sources. Omitting an extent axis preserves a grid-stride kernel's worker grid.
    The bridge supports Warp's rank-specific launch_bounds_t ABI, one-dimensional
    CUDA blocks, and four int32 scalar parameters per kernel. Unsupported bindings
    fail during preparation. Invalid sources disable their nodes and report errors;
    the caller owns domain-wide validation and gates unbound work too.
    """

    def __init__(self, count, *, maximum: int, capacity_nodes: int, library=None):
        if (
            count.shape != (1,)
            or count.dtype != wp.int32
            or not count.device.is_cuda
            or not count.is_contiguous
            or not getattr(count, "ptr", None)
        ):
            raise ValueError("Count must be one contiguous CUDA int32 scalar")
        if type(maximum) is not int or not 1 <= maximum < 2**31:
            raise ValueError("Prepared enable-count maximum must be positive int32")
        if type(capacity_nodes) is not int or capacity_nodes < 1:
            raise ValueError("Descriptor capacity must be positive")
        self.count, self.maximum, self.device, self.capacity_nodes = count, maximum, count.device, capacity_nodes
        self._graph = None
        self._capture_owner = None
        self._captured_nodes = {}
        self._instantiation_attempted = False
        self._library = self._load_library(library)
        self.binding_bytes = self._library.binding_size()
        self.bindings = wp.zeros(capacity_nodes * self.binding_bytes, dtype=wp.uint8, device=self.device)
        self.binding_count = wp.zeros(1, dtype=int, device=self.device)
        self.errors = wp.zeros(capacity_nodes, dtype=int, device=self.device)

    def _load_library(self, library):
        source = Path(__file__).with_suffix(".cu")
        if library is None:
            library = os.environ.get("NEWTON_CUDA_GRAPH_LIBRARY")
        if library is None:
            import fcntl  # noqa: PLC0415 - Linux-only compilation stays lazy for optional imports.

            fingerprint = hashlib.sha256(source.read_bytes() + str(self.device.arch).encode()).hexdigest()[:16]
            cache = Path(
                wp.config.kernel_cache_dir or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "warp"
            )
            cache = cache / "newton_cuda_graph"
            cache.mkdir(parents=True, exist_ok=True)
            library = cache / f"cuda_graph_{fingerprint}.so"
            with (cache / f"{fingerprint}.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not library.exists():
                    toolkit = Path(os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "/usr/local/cuda")))
                    compiler = os.environ.get("CUDACXX") or shutil.which("nvcc") or str(toolkit / "bin" / "nvcc")
                    if not Path(compiler).is_file():
                        raise RuntimeError(
                            "MuJoCoWorlds needs a CUDA toolkit to prepare its graph bridge; provide nvcc "
                            "through CUDACXX/CUDA_HOME or a prepared NEWTON_CUDA_GRAPH_LIBRARY"
                        )
                    temporary = cache / f"{fingerprint}.{os.getpid()}.tmp.so"
                    command = [
                        compiler,
                        "-std=c++17",
                        f"-arch=sm_{self.device.arch}",
                        "-shared",
                        "-Xcompiler",
                        "-fPIC",
                        "-rdc=true",
                        str(source),
                        "-o",
                        str(temporary),
                        "-lcuda",
                    ]
                    try:
                        subprocess.run(command, check=True, capture_output=True, text=True)
                        temporary.replace(library)
                    except subprocess.CalledProcessError as error:
                        raise RuntimeError(
                            f"CUDA bridge compilation failed ({error.returncode}): {' '.join(command)}\n"
                            f"{error.stdout or ''}{error.stderr or ''}"
                        ) from error
                    finally:
                        temporary.unlink(missing_ok=True)
        lib = ct.CDLL(str(Path(library).resolve()))
        pointer = ct.c_void_p
        lib.binding_size.restype = ct.c_size_t
        lib.capture_tail.argtypes = [pointer, ct.POINTER(pointer), ct.POINTER(pointer)]
        lib.validate_node_owner.argtypes = [pointer, pointer]
        lib.prepare_binding.argtypes = [
            pointer,
            ct.c_int,
            ct.c_int,
            pointer,
            ct.POINTER(ct.c_int),
            ct.POINTER(pointer),
            ct.POINTER(ct.c_int),
            ct.c_int,
            pointer,
        ]
        lib.launch_update.argtypes = [pointer, pointer, pointer, pointer, ct.c_int, pointer, ct.c_int]
        lib.instantiate_upload.argtypes = [pointer, pointer, ct.c_uint64, ct.POINTER(pointer)]
        return lib

    def capture_tail(self, stream=None) -> int:
        """Get the exact just-recorded kernel node, including inside a conditional body."""
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Capture stream must use the count device")
        owner = self.device.captures.get(stream)
        if getattr(owner, "_parallel_capture_failed", False):
            raise RuntimeError("Discard this failed parallel capture")
        if self._capture_owner is None or self._capture_owner() is not owner or owner is None:
            raise RuntimeError("Capture nodes only after this owner's updater in the same Warp graph")
        if self._graph is not None:
            raise RuntimeError("Cannot add nodes after binding preparation")
        node, graph = ct.c_void_p(), ct.c_void_p()
        _check(self._library.capture_tail(stream.cuda_stream, ct.byref(node), ct.byref(graph)), "capture kernel tail")
        self._captured_nodes[node.value] = graph.value
        return node.value

    def capture_update(self, stream=None):
        """Record the GPU updater before its bound consumers; allocate nothing during capture."""
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Update stream must use the count device")
        owner = self.device.captures.get(stream)
        if getattr(owner, "_parallel_capture_failed", False):
            raise RuntimeError("Discard this failed parallel capture")
        if owner is None:
            raise RuntimeError("Record updates inside a Warp-managed graph capture")
        if self._capture_owner is not None:
            raise RuntimeError("Record this owner's updater exactly once")
        _check(
            self._library.launch_update(
                stream.cuda_stream,
                self.bindings.ptr,
                self.binding_count.ptr,
                self.count.ptr,
                self.maximum,
                self.errors.ptr,
                self.capacity_nodes,
            ),
            "record GPU updates",
        )
        # Warp retains this same Graph object while swapping its native handle for
        # conditional bodies. Retain identity, not the changing capture ID/handle.
        self._capture_owner = weakref.ref(owner)

    def bind(self, graph, bindings):
        """Mark and bind exact semantic nodes before first instantiation; discard graph if this fails."""
        if getattr(graph, "_parallel_capture_failed", False):
            raise RuntimeError("Discard this failed parallel capture")
        bindings = tuple(bindings)
        if self._graph is not None or graph.graph_exec is not None:
            raise RuntimeError("Bindings require a fresh graph and can be prepared only once")
        if graph.device != self.device:
            raise ValueError("Graph and count must share a device")
        if self._capture_owner is None or self._capture_owner() is not graph:
            raise ValueError("Bindings must belong to the graph that captured this owner's updater")
        if not 1 <= len(bindings) <= self.capacity_nodes or len({b.node for b in bindings}) != len(bindings):
            raise ValueError("Bindings must fit capacity and identify distinct nodes")
        claims = getattr(graph, "device_node_owners", {})
        if any(binding.node in claims for binding in bindings):
            raise ValueError("A CUDA node already has a device update owner")
        for binding in bindings:
            if type(binding.node) is not int or binding.node not in self._captured_nodes:
                raise ValueError("Bindings must use exact nodes recorded by this owner's capture_tail")
            if type(binding.rank) is not int or not 1 <= binding.rank <= 4:
                raise ValueError("Invalid Warp launch rank")
            if binding.extent_axis is not None and (
                type(binding.extent_axis) is not int or not 0 <= binding.extent_axis < binding.rank
            ):
                raise ValueError("Invalid Warp launch extent axis")
            if binding.extent_axis is None and binding.extent_source is not None:
                raise ValueError("An unchanged launch cannot supply an extent source")
            if binding.extent_source is not None:
                self._validate_source(binding.extent_source)
            if len(binding.parameters) > 4 or any(not isinstance(p, Int32Parameter) for p in binding.parameters):
                raise ValueError("Only up to four explicit int32 count parameters are supported")
            indices = [p.index for p in binding.parameters]
            if any(type(i) is not int or i < 1 for i in indices) or len(set(indices)) != len(indices):
                raise ValueError("Scalar parameter indices must be distinct positive integers")
            for parameter in binding.parameters:
                if type(parameter.maximum) is not int or not 0 <= parameter.maximum < 2**31:
                    raise ValueError("Scalar parameter maximum must be nonnegative int32")
                self._validate_source(parameter.source)
            _check(
                self._library.validate_node_owner(binding.node, self._captured_nodes[binding.node]),
                "validate captured node owner",
            )
        host = ct.create_string_buffer(self.capacity_nodes * self.binding_bytes)
        # Marking nodes is irreversible, so retain preparation ownership even if a
        # later binding fails; this owner cannot be reused with another graph.
        self._graph = weakref.ref(graph)
        graph.device_node_owners = claims
        claims.update((binding.node, self) for binding in bindings)
        sources = tuple(p.source for binding in bindings for p in binding.parameters)
        sources += tuple(binding.extent_source for binding in bindings if binding.extent_source is not None)
        graph.world_owners = (*getattr(graph, "world_owners", ()), self, self.count, *sources)
        for ordinal, binding in enumerate(bindings):
            parameters = binding.parameters
            scalars = (ct.c_int * len(parameters))(*(p.index for p in parameters))
            pointers = (ct.c_void_p * len(parameters))(*(p.source.ptr for p in parameters))
            maxima = (ct.c_int * len(parameters))(*(p.maximum for p in parameters))
            axis = -1 if binding.extent_axis is None else binding.extent_axis
            extent = binding.extent_source if binding.extent_source is not None else self.count
            _check(
                self._library.prepare_binding(
                    binding.node,
                    binding.rank,
                    axis,
                    extent.ptr,
                    scalars,
                    pointers,
                    maxima,
                    len(parameters),
                    ct.byref(host, ordinal * self.binding_bytes),
                ),
                f"prepare binding {ordinal}",
            )
        self.bindings.assign(np.frombuffer(host.raw, dtype=np.uint8))
        self.binding_count.fill_(len(bindings))
        self.prepared_bindings = bindings

    def _validate_source(self, source):
        if (
            getattr(source, "shape", None) != (1,)
            or getattr(source, "dtype", None) != wp.int32
            or getattr(source, "device", None) != self.device
            or not getattr(source, "is_contiguous", False)
            or not getattr(source, "ptr", None)
        ):
            raise ValueError("Each count source must be one contiguous int32 scalar on the binding device")

    def instantiate_upload(self, graph, *, stream=None, flags=1):
        """Instantiate once and upload explicitly; flags=1 matches Warp AutoFreeOnLaunch."""
        if getattr(graph, "_parallel_capture_failed", False):
            raise RuntimeError("Discard this failed parallel capture")
        if self._graph is None or self._graph() is not graph or not hasattr(self, "prepared_bindings"):
            raise RuntimeError("Graph must have a successful binding preparation")
        if graph.graph_exec is not None or self._instantiation_attempted:
            raise RuntimeError("A graph with device-updatable nodes can be instantiated only once")
        stream = wp.get_stream(self.device) if stream is None else stream
        if stream.device != self.device:
            raise ValueError("Instantiation stream must use the count device")
        if type(flags) is not int or not 0 <= flags < 2**64:
            raise ValueError("Instantiation flags must be an unsigned 64-bit integer")
        executable = ct.c_void_p()
        self._instantiation_attempted = True
        _check(
            self._library.instantiate_upload(graph.graph, stream.cuda_stream, flags, ct.byref(executable)),
            "instantiate/upload graph",
        )
        # The experiment bridges a missing public Warp explicit-instantiation API.
        # Normal Warp Graph destruction owns both returned CUDA handles.
        graph.graph_exec = executable

    def memory_report(self):
        return {
            "binding_capacity": self.capacity_nodes,
            "binding_bytes": self.binding_bytes,
            "enable_count_maximum": self.maximum,
            "prepared_nodes": len(getattr(self, "prepared_bindings", ())),
            "device_payload_bytes": self.bindings.capacity + self.binding_count.capacity + self.errors.capacity,
            "scope": "update table only; graph execution storage, modules and count owner excluded",
        }
