// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

// Mechanical CUDA device-node binding. No world or physics ownership.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <vector>

struct Binding {
    CUgraphDeviceNode node;
    size_t bounds_offset, bounds_size, work_per_item;
    size_t scalar_offsets[4];
    const int *extent_count, *scalar_sources[4];
    int scalar_maxima[4];
    int scalar_count, shape_offset, size_offset, block_dim, maximum;
    unsigned int grid_limit_x, grid_limit_y, grid_limit_z;
    alignas(8) unsigned char bounds[32];
};

__global__ void update_nodes(const Binding *bindings, const int *binding_count, const int *count,
                             int maximum_count, int *errors) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= binding_count[0]) return;
    const Binding &binding = bindings[i];
    int live = count[0];
    int extent = binding.shape_offset >= 0 ? binding.extent_count[0] : 1;
    int parameters[4];
    bool valid = live >= 0 && live <= maximum_count && extent >= 0
        && (binding.shape_offset < 0 || extent <= binding.maximum);
    for (int arg = 0; arg < binding.scalar_count; ++arg) {
        parameters[arg] = binding.scalar_sources[arg][0];
        valid = valid && parameters[arg] >= 0 && parameters[arg] <= binding.scalar_maxima[arg];
    }
    auto node = reinterpret_cast<cudaGraphDeviceNode_t>(binding.node);
    // Disabled nodes need no bounds or parameter writes; the next active replay writes them all.
    if (!valid || live == 0 || extent == 0) {
        cudaError_t error = cudaGraphKernelNodeSetEnabled(node, false);
        errors[i] = error == cudaSuccess ? (valid ? 0 : -1) : int(error);
        return;
    }
    cudaError_t error = cudaSuccess;
    if (binding.shape_offset >= 0) {
        alignas(8) unsigned char bounds[32];
        for (int b = 0; b < 32; ++b) bounds[b] = binding.bounds[b];
        *reinterpret_cast<int *>(bounds + binding.shape_offset) = extent;
        size_t work = size_t(extent) * binding.work_per_item;
        *reinterpret_cast<size_t *>(bounds + binding.size_offset) = work;
        error = cudaGraphKernelNodeSetParam(node, binding.bounds_offset, bounds, binding.bounds_size);
        if (error == cudaSuccess) {
            size_t blocks = (work + binding.block_dim - 1) / binding.block_dim;
            // Warp's lean kernels spill a large linear launch into Y/Z. Keep
            // the captured envelope, including 1D caps for grid-stride kernels.
            unsigned int x = unsigned(blocks < binding.grid_limit_x ? blocks : binding.grid_limit_x);
            if (!x) x = 1;
            size_t remaining = (blocks + x - 1) / x;
            unsigned int y = unsigned(remaining < binding.grid_limit_y ? remaining : binding.grid_limit_y);
            if (!y) y = 1;
            remaining = (remaining + y - 1) / y;
            unsigned int z = unsigned(remaining < binding.grid_limit_z ? remaining : binding.grid_limit_z);
            error = cudaGraphKernelNodeSetGridDim(node, dim3(x, y, z ? z : 1));
        }
    }
    for (int arg = 0; arg < binding.scalar_count && error == cudaSuccess; ++arg)
        error = cudaGraphKernelNodeSetParam(node, binding.scalar_offsets[arg], parameters[arg]);
    if (error == cudaSuccess) error = cudaGraphKernelNodeSetEnabled(node, true);
    else cudaGraphKernelNodeSetEnabled(node, false);  // Preserve the original error; caller quarantines failed work.
    errors[i] = int(error);
}

extern "C" size_t binding_size() { return sizeof(Binding); }

extern "C" int get_last_kernel_node(void *stream, void **node, void **graph) {
    CUstreamCaptureStatus status;
    const CUgraphNode *dependencies;
    size_t count;
    CUgraph captured;
    CUresult error = cuStreamGetCaptureInfo_v3(reinterpret_cast<CUstream>(stream), &status, nullptr, &captured,
                                           &dependencies, nullptr, &count);
    if (error) return int(error);
    if (status != CU_STREAM_CAPTURE_STATUS_ACTIVE || count != 1) return -100;
    CUgraphNodeType type;
    error = cuGraphNodeGetType(dependencies[0], &type);
    if (error) return int(error);
    if (type != CU_GRAPH_NODE_TYPE_KERNEL) return -101;
    *node = reinterpret_cast<void *>(dependencies[0]);
    *graph = reinterpret_cast<void *>(captured);
    return 0;
}

extern "C" int validate_node_owner(void *handle, void *graph) {
    size_t count = 0;
    CUgraph captured = reinterpret_cast<CUgraph>(graph);
    CUresult error = cuGraphGetNodes(captured, nullptr, &count);
    if (error) return int(error);
    std::vector<CUgraphNode> nodes(count);
    error = cuGraphGetNodes(captured, nodes.data(), &count);
    if (error) return int(error);
    for (CUgraphNode node : nodes) if (node == reinterpret_cast<CUgraphNode>(handle)) return 0;
    return -112;
}

extern "C" int prepare_binding(void *handle, int rank, int axis, void *extent_count, const int *scalar_args,
                                const void *const *scalar_sources, const int *scalar_maxima,
                                int scalar_count, void *output) {
    if (rank < 1 || rank > 4 || axis < -1 || axis >= rank || scalar_count < 0 || scalar_count > 4) return -102;
    CUgraphNode node = reinterpret_cast<CUgraphNode>(handle);
    CUDA_KERNEL_NODE_PARAMS params = {};
    CUresult error = cuGraphKernelNodeGetParams(node, &params);
    if (error) return int(error);
    Binding binding = {};
    error = cuFuncGetParamInfo(params.func, 0, &binding.bounds_offset, &binding.bounds_size);
    if (error) return int(error);
    binding.size_offset = ((rank * int(sizeof(int)) + 7) / 8) * 8;
    if (binding.bounds_size != binding.size_offset + 2 * sizeof(size_t) || binding.bounds_size > sizeof(binding.bounds))
        return -103;
    if (!params.kernelParams || !params.kernelParams[0]) return -104;
    std::memcpy(binding.bounds, params.kernelParams[0], binding.bounds_size);
    binding.shape_offset = axis < 0 ? -1 : axis * int(sizeof(int));
    binding.extent_count = static_cast<const int *>(extent_count);
    if (axis >= 0) {
        if (!extent_count) return -114;
        binding.maximum = *reinterpret_cast<int *>(binding.bounds + binding.shape_offset);
        size_t work = *reinterpret_cast<size_t *>(binding.bounds + binding.size_offset);
        if (binding.maximum <= 0 || work % binding.maximum) return -105;
        binding.work_per_item = work / binding.maximum;
    }
    binding.block_dim = params.blockDimX;
    binding.grid_limit_x = params.gridDimX;
    binding.grid_limit_y = params.gridDimY;
    binding.grid_limit_z = params.gridDimZ;
    if (params.blockDimX == 0 || params.blockDimY != 1 || params.blockDimZ != 1 || params.gridDimX == 0
        || params.gridDimY == 0 || params.gridDimZ == 0) return -106;
    binding.scalar_count = scalar_count;
    for (int i = 0; i < scalar_count; ++i) {
        if (scalar_args[i] <= 0) return -107;
        if (!scalar_sources[i] || scalar_maxima[i] < 0) return -114;
        for (int j = 0; j < i; ++j) if (scalar_args[i] == scalar_args[j]) return -113;
        size_t size;
        error = cuFuncGetParamInfo(params.func, scalar_args[i], &binding.scalar_offsets[i], &size);
        if (error) return int(error);
        if (size != sizeof(int)) return -108;
        binding.scalar_sources[i] = static_cast<const int *>(scalar_sources[i]);
        binding.scalar_maxima[i] = scalar_maxima[i];
    }
    CUkernelNodeAttrValue attr = {};
    attr.deviceUpdatableKernelNode.deviceUpdatable = 1;
    error = cuGraphKernelNodeSetAttribute(node, CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE, &attr);
    if (error) return int(error);
    error = cuGraphKernelNodeGetAttribute(node, CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE, &attr);
    if (error) return int(error);
    binding.node = attr.deviceUpdatableKernelNode.devNode;
    if (!binding.node) return -109;
    std::memcpy(output, &binding, sizeof(binding));
    return 0;
}

extern "C" int launch_update(void *stream, void *bindings, void *binding_count, void *count,
                              int maximum_count, void *errors, int capacity) {
    if (capacity <= 0 || maximum_count <= 0) return -110;
    update_nodes<<<(capacity + 127) / 128, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const Binding *>(bindings), static_cast<const int *>(binding_count),
        static_cast<const int *>(count), maximum_count, static_cast<int *>(errors));
    return int(cudaGetLastError());
}

extern "C" int instantiate_and_upload(void *graph, void *stream, unsigned long long flags, void **executable) {
    CUgraphExec exec;
    CUresult error = cuGraphInstantiateWithFlags(&exec, reinterpret_cast<CUgraph>(graph), flags);
    if (error) return int(error);
    error = cuGraphUpload(exec, reinterpret_cast<CUstream>(stream));
    if (error) { cuGraphExecDestroy(exec); return int(error); }
    *executable = reinterpret_cast<void *>(exec);
    return 0;
}
