# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run native lane-group sweep bodies on the CPU, one thread per CUDA thread, to compare them bit for bit.

A native sweep body runs in lane groups (:func:`newton._src.solvers.feather_pgs.device_preludes.lane_group`):
the lanes of a group synchronize, shuffle and vote among themselves and share their block's shared memory.
:class:`SimtProgram` compiles bodies that share a native ABI into one host program (``g++``) under an emulation of
those intrinsics: each lane of a world's group is a thread with its own ``threadIdx``, ``__shared__`` storage is
static, and every shuffle, vote and ``__syncwarp`` is an exchange among the group's threads at a barrier. The
groups' reduction trees and lane layouts are the device's, so bodies whose floating-point dataflows agree give
identical bits (contraction into fused multiply-adds is off for both). The projection laws and Warp's vector types
come from Warp's native headers, as Warp's CPU kernels take them.

Lanes run freely between barriers here, where a device warp runs in lockstep: a body whose lanes race on shared
memory (one lane overwriting what the others may not have read yet) can produce other values here than on the
device.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np
import warp as wp

HEADER = r"""
#pragma once
#include "builtin.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using std::isfinite;

struct Dim3 {
    unsigned x = 0, y = 0, z = 0;
};
inline thread_local Dim3 threadIdx;
inline Dim3 blockDim{64, 1, 1};

#define __device__
#define __forceinline__ inline
#define __shared__ static

inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }

// float4 comes with Warp's tile header on the CPU.
struct alignas(16) int4 {
    int x, y, z, w;
};
inline int4 make_int4(int x, int y, int z, int w) { return int4{x, y, z, w}; }

inline float __int_as_float(int value) { float result; std::memcpy(&result, &value, 4); return result; }
inline int __float_as_int(float value) { int result; std::memcpy(&result, &value, 4); return result; }
inline int __popc(unsigned bits) { return __builtin_popcount(bits); }
inline int __popcll(unsigned long long bits) { return __builtin_popcountll(bits); }
inline int __ffs(unsigned bits) { return __builtin_ffs(static_cast<int>(bits)); }
inline int __ffsll(unsigned long long bits) { return __builtin_ffsll(static_cast<long long>(bits)); }
inline int atomicAdd(int* address, int value) { return __atomic_fetch_add(address, value, __ATOMIC_SEQ_CST); }
inline unsigned long long atomicAdd(unsigned long long* address, unsigned long long value)
{
    return __atomic_fetch_add(address, value, __ATOMIC_SEQ_CST);
}

namespace simt {

constexpr int BLOCK = 1024;

// A reusable barrier of the lanes of one mask (a group's lanes are aligned and contiguous in their warp): a short
// spin, then a wait, so that waiting lanes burn no CPU.
struct Barrier {
    std::atomic<int> arrived{0};
    std::atomic<int> generation{0};
    void wait(int lanes)
    {
        const int seen = generation.load(std::memory_order_acquire);
        if (arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == lanes) {
            arrived.store(0, std::memory_order_relaxed);
            generation.store(seen + 1, std::memory_order_release);
#if defined(__cpp_lib_atomic_wait)
            generation.notify_all();
#endif
            return;
        }
        for (int spin = 0; spin < 128; ++spin)
            if (generation.load(std::memory_order_acquire) != seen)
                return;
        while (generation.load(std::memory_order_acquire) == seen) {
#if defined(__cpp_lib_atomic_wait)
            generation.wait(seen, std::memory_order_acquire);
#else
            std::this_thread::yield();
#endif
        }
    }
};

inline Barrier barriers[BLOCK];
// Two banks of exchange slots, alternated by every lane's exchanges: a lane writes a bank again only after the next
// exchange's barrier, which every lane of its group reaches after reading the bank, so one barrier per exchange.
inline uint64_t slots[2][BLOCK];
inline thread_local int bank = 0;

inline void sync_mask(unsigned mask)
{
    const int warp = static_cast<int>(threadIdx.x) & ~31;
    barriers[warp + __builtin_ctz(mask)].wait(__builtin_popcount(mask));
}

inline uint64_t* publish(unsigned mask, uint64_t bits)
{
    uint64_t* current = slots[bank];
    bank ^= 1;
    current[threadIdx.x] = bits;
    sync_mask(mask);
    return current + (static_cast<int>(threadIdx.x) & ~31);
}

template <class T> inline T exchange(unsigned mask, T value, int source_lane)
{
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(T));
    bits = publish(mask, bits)[source_lane];
    T result;
    std::memcpy(&result, &bits, sizeof(T));
    return result;
}

}  // namespace simt

inline void __syncwarp(unsigned mask = 0xffffffffu) { simt::sync_mask(mask); }
inline void __syncthreads() { std::abort(); }
inline int __syncthreads_and(int) { std::abort(); }
inline int __syncthreads_or(int) { std::abort(); }

template <class T> inline T __shfl_xor_sync(unsigned mask, T value, int offset, int width = 32)
{
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int base = lane & ~(width - 1);
    return simt::exchange(mask, value, base + (((lane - base) ^ offset) % width));
}
template <class T> inline T __shfl_sync(unsigned mask, T value, int source, int width = 32)
{
    const int lane = static_cast<int>(threadIdx.x) & 31;
    return simt::exchange(mask, value, (lane & ~(width - 1)) + (source % width));
}
template <class T> inline T __shfl_up_sync(unsigned mask, T value, unsigned delta, int width = 32)
{
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int base = lane & ~(width - 1);
    return simt::exchange(mask, value, lane - base >= static_cast<int>(delta) ? lane - static_cast<int>(delta) : lane);
}
inline unsigned __ballot_sync(unsigned mask, int predicate)
{
    const uint64_t* votes = simt::publish(mask, predicate ? 1u : 0u);
    unsigned bits = 0;
    for (int lane = 0; lane < 32; ++lane)
        if (((mask >> lane) & 1u) && votes[lane])
            bits |= 1u << lane;
    return bits;
}
inline int __all_sync(unsigned mask, int predicate) { return (__ballot_sync(mask, predicate) & mask) == mask; }
inline int __any_sync(unsigned mask, int predicate) { return (__ballot_sync(mask, predicate) & mask) != 0u; }

namespace simt {

// Run body(world) for every world, one lane group at a time: its lanes are the threads of its slot in a block of
// block threads (lane groups of a block share nothing but their slots of the block's shared memory).
template <class Body> void launch(int world_count, int lanes, int block, Body body)
{
    const int groups = block / lanes;
    for (int world = 0; world < world_count; ++world) {
        std::vector<std::thread> threads;
        threads.reserve(lanes);
        for (int lane = 0; lane < lanes; ++lane)
            threads.emplace_back([&, lane] {
                threadIdx.x = static_cast<unsigned>((world % groups) * lanes + lane);
                body(world);
            });
        for (auto& thread : threads)
            thread.join();
    }
}

struct Output {
    std::string name;
    const void* data;
    size_t bytes;
};
inline std::vector<Output> outputs;

template <class T> wp::array_t<T> load(const std::string& directory, const std::string& name, int ndim)
{
    int dims[4] = {1, 1, 1, 1};
    FILE* shape = std::fopen((directory + "/" + name + ".shape").c_str(), "rb");
    if (!shape || std::fread(dims, sizeof(int), ndim, shape) != static_cast<size_t>(ndim))
        std::exit(3);
    std::fclose(shape);
    size_t count = 1;
    for (int axis = 0; axis < ndim; ++axis)
        count *= static_cast<size_t>(dims[axis]);
    auto* data = new std::vector<T>(count);
    FILE* file = std::fopen((directory + "/" + name + ".bin").c_str(), "rb");
    if (!file || std::fread(data->data(), sizeof(T), count, file) != count)
        std::exit(4);
    std::fclose(file);
    outputs.push_back({name, data->data(), count * sizeof(T)});
    if (ndim == 1)
        return wp::array_t<T>(data->data(), dims[0]);
    if (ndim == 2)
        return wp::array_t<T>(data->data(), dims[0], dims[1]);
    if (ndim == 3)
        return wp::array_t<T>(data->data(), dims[0], dims[1], dims[2]);
    return wp::array_t<T>(data->data(), dims[0], dims[1], dims[2], dims[3]);
}

inline void store(const std::string& directory)
{
    for (const Output& output : outputs) {
        FILE* file = std::fopen((directory + "/" + output.name + ".out").c_str(), "wb");
        std::fwrite(output.data, 1, output.bytes, file);
        std::fclose(file);
    }
}

}  // namespace simt

#define __CUDA_ARCH__ 1200
"""

_ELEMENTS = {
    wp.int32: ("int", np.int32),
    wp.uint32: ("unsigned", np.uint32),
    wp.int64: ("long long", np.int64),
    wp.float32: ("float", np.float32),
}


def compiler() -> str | None:
    """The host C++ compiler the emulation uses (``g++``), or ``None``."""
    return shutil.which("g++")


def _scalar(annotation) -> str | None:
    if annotation in (int, wp.int32):
        return "int"
    if annotation in (float, wp.float32):
        return "float"
    return None


def _is_array(annotation) -> bool:
    return getattr(annotation, "dtype", None) in _ELEMENTS and getattr(annotation, "ndim", None) is not None


class SimtProgram:
    """Native bodies of one ABI, compiled into a host program that runs them under the lane emulation.

    Args:
        bodies: ``{label: KernelTemplate}`` of bodies with the same native parameters, the first of them the world.
        lanes: Lanes per world (the bodies' lane group).
        block: Threads per block of the bodies' launches.
        directory: Where to write the program and its runs.
    """

    def __init__(self, bodies: dict, lanes: int, block: int, directory: str):
        self.lanes, self.block, self.directory = int(lanes), int(block), directory
        templates = list(bodies.values())
        self.params = templates[0].native_params
        for template in templates[1:]:
            if [name for name, _ in template.native_params] != [name for name, _ in self.params]:
                raise ValueError("bodies need the same native parameters")
        self.labels = tuple(bodies)
        structs, loads, arguments, self.arrays, self.scalars = [], [], [], {}, []
        for index, (name, annotation) in enumerate(self.params):
            if index == 0:
                arguments.append("world")
                continue
            scalar = _scalar(annotation)
            if scalar is not None:
                loads.append(
                    f"    const {scalar} {name} = static_cast<{scalar}>(program_scalars[{len(self.scalars)}]);"
                )
                self.scalars.append(name)
            elif _is_array(annotation):
                element = _ELEMENTS[annotation.dtype][0]
                loads.append(
                    f'    const auto {name} = simt::load<{element}>(program_directory, "{name}", {annotation.ndim});'
                )
                self.arrays[name] = annotation
            else:
                fields = []
                values = []
                for field, var in annotation.vars.items():
                    path = f"{name}.{field}"
                    field_scalar = _scalar(var.type)
                    if field_scalar is not None:
                        fields.append(f"    {field_scalar} {field};")
                        values.append(f"static_cast<{field_scalar}>(program_scalars[{len(self.scalars)}])")
                        self.scalars.append(path)
                    else:
                        element = _ELEMENTS[var.type.dtype][0]
                        fields.append(f"    wp::array_t<{element}> {field};")
                        values.append(f'simt::load<{element}>(program_directory, "{path}", {var.type.ndim})')
                        self.arrays[path] = var.type
                struct = f"Struct_{name}"
                structs.append(f"struct {struct} {{\n" + "\n".join(fields) + "\n};")
                loads.append(f"    const {struct} {name}{{{', '.join(values)}}};")
            arguments.append(name)
        signature = ", ".join(
            f"int {name}" if index == 0 else f"{self._ctype(name, annotation)} {name}"
            for index, (name, annotation) in enumerate(self.params)
        )
        functions = [
            f"void body_{index}({signature}) {{\n{template.snippet}\n}}" for index, template in enumerate(templates)
        ]
        calls = "\n".join(
            f"        if (program_body == {index}) body_{index}({', '.join(arguments)});"
            for index in range(len(templates))
        )
        source = "\n".join(
            [
                '#include "simt.h"',
                *structs,
                *functions,
                "int main(int argc, char** argv) {",
                "    const std::string program_directory = argv[1];",
                "    const int program_body = std::atoi(argv[2]), program_worlds = std::atoi(argv[3]);",
                f"    std::vector<double> program_scalars({max(len(self.scalars), 1)});",
                '    FILE* program_file = std::fopen((program_directory + "/scalars.bin").c_str(), "rb");',
                f"    if (!program_file || std::fread(program_scalars.data(), sizeof(double), {len(self.scalars)}, "
                f"program_file) != {len(self.scalars)}) return 2;",
                "    std::fclose(program_file);",
                *loads,
                f"    simt::launch(program_worlds, {self.lanes}, {self.block}, [&](int world) {{",
                calls,
                "    });",
                "    simt::store(program_directory);",
                "    return 0;",
                "}",
            ]
        )
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "simt.h"), "w") as header:
            header.write(HEADER)
        program = os.path.join(directory, "program.cpp")
        with open(program, "w") as file:
            file.write(source)
        self.binary = os.path.join(directory, "program")
        native = os.path.join(os.path.dirname(wp.__file__), "native")
        command = [compiler() or "g++", "-std=c++20", "-O1", "-ffp-contract=off", "-pthread", "-w"]
        command += [f"-I{directory}", f"-I{native}", program, "-o", self.binary]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"the emulation program does not compile:\n{result.stderr[-6000:]}")

    @staticmethod
    def _ctype(name, annotation) -> str:
        scalar = _scalar(annotation)
        if scalar is not None:
            return scalar
        if _is_array(annotation):
            return f"wp::array_t<{_ELEMENTS[annotation.dtype][0]}>"
        return f"Struct_{name}"

    def run(self, label: str, world_count: int, arrays: dict, scalars: dict) -> dict:
        """Run body ``label`` over ``world_count`` worlds; return every array parameter after the run.

        Arrays absent from ``arrays`` (including struct fields, as ``"param.field"``) are zeros of one element per
        axis; absent scalars are zero.
        """
        run = tempfile.mkdtemp(dir=self.directory)
        inputs = {}
        for path, annotation in self.arrays.items():
            dtype = _ELEMENTS[annotation.dtype][1]
            value = np.ascontiguousarray(arrays.get(path, np.zeros((1,) * annotation.ndim)), dtype=dtype)
            if value.ndim != annotation.ndim:
                raise ValueError(f"{path} needs {annotation.ndim} axes, got {value.shape}")
            inputs[path] = value
            value.tofile(os.path.join(run, f"{path}.bin"))
            np.asarray(value.shape, dtype=np.int32).tofile(os.path.join(run, f"{path}.shape"))
        np.asarray([float(scalars.get(name, 0)) for name in self.scalars], dtype=np.float64).tofile(
            os.path.join(run, "scalars.bin")
        )
        result = subprocess.run(
            [self.binary, run, str(self.labels.index(label)), str(int(world_count))],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        if result.returncode:
            raise RuntimeError(f"body {label} failed ({result.returncode}): {result.stderr[-2000:]}")
        return {
            path: np.fromfile(os.path.join(run, f"{path}.out"), dtype=value.dtype).reshape(value.shape)
            for path, value in inputs.items()
        }
