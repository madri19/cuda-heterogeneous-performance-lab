# Phase 3 Summary - CUDA Heterogeneous Performance Lab

Phase 3 explored CUDA, GPU execution, CPU/GPU workload partitioning, memory transfer overhead, and NVIDIA profiling tools.

This phase builds on:

- Phase 1: ARM Embedded Linux fundamentals
- Phase 2: Linux systems performance on x86_64 and ARM server hardware

Phase 3 moves from CPU-side systems performance into heterogeneous CPU/GPU performance.

## Platform

- GPU: NVIDIA GeForce RTX 3060 Ti
- VRAM: 8 GiB
- Compute capability: 8.6
- SM count: 38
- Warp size: 32
- Max threads per block: 1024
- CUDA Toolkit: 13.2
- Driver: NVIDIA 596.49
- WSL Ubuntu 26.04

## Experiments

| Experiment | Focus |
|---|---|
| 01_cuda_device_info | CUDA runtime, driver, and device properties |
| 02_vector_add_baseline | First CUDA kernel and correctness validation |
| 03_cpu_vs_gpu_vector_add | CPU vs GPU kernel-only vector-add comparison |
| 04_memory_transfer_overhead | Host/device transfer cost versus kernel time |
| 05_cuda_memory_bandwidth | Device-only global memory bandwidth |
| 06_grid_block_scaling | Block-size effects on memory-parallel kernels |
| 07_shared_memory_tiling | Shared-memory tiled transpose versus naive transpose |
| 08_streams_overlap | Pageable-memory CUDA stream overlap attempt |
| 08b_pinned_memory_overlap | Pinned-memory transfer behavior and stream overlap |
| 09_cuda_reduction | Shared-memory block-level reduction |
| 10_nsight_profile_analysis | Nsight Systems and Nsight Compute profiling |

## Main Results

### Device Baseline

The local CUDA environment successfully detected one CUDA-capable GPU:

    NVIDIA GeForce RTX 3060 Ti
    Compute capability: 8.6
    Global memory: 8191 MiB
    Multiprocessors: 38

CUDA runtime and driver both reported CUDA 13.2 compatibility.

### Vector Add Baseline

The first vector-add run showed a slow cold-start timing, but warm runs stabilized around 2.2 ms.

Key lesson:

    Do not trust single first-run CUDA timings.

Warmup launches are required before meaningful measurement.

### CPU vs GPU Vector Add

For 16,777,216 float elements:

| Target | Average Time | Estimated Bandwidth |
|---|---:|---:|
| CPU | 8.84773 ms | 21.1919 GiB/sec |
| GPU kernel | 0.483382 ms | 387.892 GiB/sec |

Kernel-only GPU speedup:

    18.3038x

This showed that the GPU kernel was much faster when data was already resident on the device.

### Transfer Overhead

For the same vector-add workload:

| Measurement | Average Time |
|---|---:|
| CPU | 9.32453 ms |
| GPU H2D | 12.7318 ms |
| GPU kernel | 0.509534 ms |
| GPU D2H | 6.8729 ms |
| GPU end-to-end | 20.2527 ms |

Kernel-only GPU speedup:

    18.3001x

End-to-end GPU speedup:

    0.46041x

Transfer share of GPU time:

    96.8005%

Key lesson:

    GPU kernel speed does not guarantee end-to-end acceleration.

For transfer-heavy workloads, host/device movement can dominate runtime.

### Device Memory Bandwidth

Best observed device-only copy bandwidth:

    about 386 GiB/sec

128 and 256 threads per block performed best for the copy workload.

### Grid / Block Scaling

For vector add on 67,108,864 elements:

| Threads per Block | Estimated Bandwidth |
|---:|---:|
| 32 | 275.595 GiB/sec |
| 64 | 389.139 GiB/sec |
| 128 | 390.18 GiB/sec |
| 256 | 390.202 GiB/sec |
| 512 | 390.092 GiB/sec |
| 1024 | 380.246 GiB/sec |

Best observed configuration:

    256 threads/block

Key lesson:

    Maximum block size is not automatically optimal.

The stable high-performance range was 64 to 512 threads per block.

### Shared Memory Tiling

Matrix transpose results:

| Kernel | Average Time | Estimated Bandwidth |
|---|---:|---:|
| Naive transpose | 4.0959 ms | 122.073 GiB/sec |
| Shared-memory tiled transpose | 1.32939 ms | 376.112 GiB/sec |

Tiled speedup:

    3.08103x

Key lesson:

    Shared memory helps when it changes the memory access pattern.

The tiled transpose improved coalescing and avoided the worst global-memory access pattern.

### CUDA Streams

Pageable-memory stream experiment:

| Mode | Average Time |
|---|---:|
| Sequential | 19.9095 ms |
| Streamed | 20.6194 ms |

Speedup:

    0.965574x

Streams did not help with pageable memory.

### Pinned Memory

Pinned-memory stream experiment:

| Mode | Average Time |
|---|---:|
| Pinned sequential | 8.46601 ms |
| Pinned streamed | 8.73883 ms |

Pinned memory greatly improved transfer-heavy runtime compared with pageable memory, but streams still did not improve this workload.

Key lesson:

    Pinned memory helps transfer-heavy CUDA paths.
    Streams do not automatically help.

### CUDA Reduction

Reduction results:

| Target | Average Time | Estimated Read Bandwidth |
|---|---:|---:|
| CPU | 44.1059 ms | 5.66817 GiB/sec |
| GPU kernel | 1.01836 ms | 245.492 GiB/sec |

GPU kernel speedup:

    43.3106x

The GPU reduction used shared-memory block-local accumulation and produced the same result as the CPU.

### Nsight Profiling

Nsight Systems confirmed that vector-add kernels were short and stable, while memory transfers dominated end-to-end time.

Nsight Compute was initially blocked by GPU performance-counter permissions. The issue was fixed by enabling GPU performance counter access in NVIDIA Control Panel.

Nsight Compute SpeedOfLight metrics for the reduction kernel:

| Metric | Value |
|---|---:|
| Duration | 1.02 ms |
| Memory Throughput | 89.80% |
| DRAM Throughput | 60.99% |
| L1/TEX Cache Throughput | 89.90% |
| L2 Cache Throughput | 21.53% |
| Compute (SM) Throughput | 89.80% |

Nsight Compute confirmed the reduction kernel was using a high fraction of available compute or memory performance.

## Core Lessons

Phase 3 showed that CUDA performance depends on the whole CPU/GPU path, not just raw kernel speed.

Major findings:

- warmup matters
- kernel-only timing can mislead
- host/device transfer cost can dominate runtime
- pinned memory can significantly reduce transfer-heavy runtime
- streams require the right memory and workload shape
- block size should be measured, not guessed
- shared memory helps when it improves memory access patterns
- reductions require different kernel structure than simple map-style kernels
- NVIDIA profiling tools are necessary for explaining bottlenecks

## Relationship to Phase 2

Phase 2 focused on CPU-side multicore bottlenecks:

- locks
- atomics
- cache lines
- false sharing
- allocator behavior
- scheduler behavior
- queue contention

Phase 3 focused on heterogeneous bottlenecks:

- kernel launch behavior
- device memory bandwidth
- host/device transfer overhead
- pinned memory
- streams
- shared memory
- profiling CUDA kernels

Together they form a systems-performance progression:

    CPU multicore performance
    -> GPU kernel performance
    -> CPU/GPU coordination cost

## Status

Phase 3A is complete.

