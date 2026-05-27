# Experiment 10 - Nsight Profile Analysis

Uses NVIDIA Nsight Systems and Nsight Compute CLI tools to profile CUDA workloads.

## Goal

Validate manual CUDA timing results with NVIDIA profiling tools.

This experiment profiles:

- CPU vs GPU vector add
- memory transfer overhead benchmark
- CUDA reduction benchmark

## Tools

Detected tools:

    /usr/local/cuda-13.2/bin/nsys
    /usr/local/cuda-13.2/bin/ncu

Versions:

    NVIDIA Nsight Systems version 2025.6.3
    NVIDIA Nsight Compute CLI version 2026.1.1.0

## Nsight Systems - Vector Add

Profiled command:

    nsys profile --stats=true ./build/cpu_vs_gpu_vector_add

Key results:

- 25 vector-add kernel launches
- Average kernel time: about 0.482 ms
- Host-to-device copies: 2 copies, about 14.94 ms total
- Device-to-host copy: 1 copy, about 6.92 ms total

The profiler confirmed the manually measured kernel timing from earlier experiments.

## Nsight Systems - Memory Transfer Overhead

Profiled command:

    nsys profile --stats=true ./build/memory_transfer_overhead

Key results:

- `cudaMemcpy` dominated CUDA API time
- 25 vector-add kernel launches
- Average kernel time: about 0.480 ms
- Host-to-device copies: 42 transfers, about 275.1 ms total
- Device-to-host copies: 21 transfers, about 147.0 ms total

This confirms that the transfer-heavy benchmark is dominated by host/device memory movement, not kernel execution.

## Nsight Compute

Attempted command:

    ncu --set full ./build/cuda_reduction

Result:

    ERR_NVGPUCTRPERM

Nsight Compute could launch the target process, but GPU performance counter access was blocked by NVIDIA permission settings.

A lighter section set also failed with the same permission issue.

## Interpretation

Nsight Systems successfully confirmed the key Phase 3 finding:

    CUDA kernels can be much faster than CPU execution, but host/device transfer cost can dominate end-to-end runtime.

Nsight Compute requires additional GPU performance-counter permissions before detailed kernel metrics can be collected.

## Key Lesson

Manual CUDA event timing is useful, but profiling provides a stronger explanation.

For these workloads, Nsight Systems showed:

- kernel execution was short and stable
- memory transfers consumed most end-to-end GPU time
- profiling output matched earlier manual timing conclusions

## Status

Nsight Systems profiling is working.

Nsight Compute is installed but blocked by GPU performance-counter permissions.

