# CUDA Heterogeneous Performance Lab

Phase 3 performance lab focused on CUDA, GPU execution, and CPU/GPU workload partitioning.

This project builds on:

- Phase 1: ARM Embedded Linux fundamentals
- Phase 2: Linux systems performance on x86_64 and ARM server hardware

Phase 3 moves into heterogeneous performance:

    CPU multicore systems
    -> GPU acceleration
    -> CPU/GPU coordination

## Hardware

- GPU: NVIDIA GeForce RTX 3060 Ti
- VRAM: 8 GiB
- Driver: NVIDIA 596.49 on Windows / WSL
- CUDA runtime supported by driver: 13.2
- WSL Ubuntu 26.04
- CUDA Toolkit: 13.2

## Scope

Focus areas:

- CUDA device discovery
- kernel launch overhead
- CPU vs GPU throughput
- host/device memory transfer cost
- global memory bandwidth
- grid/block sizing
- shared memory
- streams and overlap
- reductions
- profiling with NVIDIA tools

## Planned Experiments

| Experiment | Focus |
|---|---|
| [01_cuda_device_info](experiments/01_cuda_device_info) | Capture CUDA device properties and runtime environment |
| [02_vector_add_baseline](experiments/02_vector_add_baseline) | First CUDA kernel and correctness check |
| [03_cpu_vs_gpu_vector_add](experiments/03_cpu_vs_gpu_vector_add) | Compare CPU and GPU vector addition throughput |
| [04_memory_transfer_overhead](experiments/04_memory_transfer_overhead) | Measure host/device transfer overhead |
| [05_cuda_memory_bandwidth](experiments/05_cuda_memory_bandwidth) | Measure GPU global memory bandwidth |
| [06_grid_block_scaling](experiments/06_grid_block_scaling) | Explore block size and grid size effects |
| [07_shared_memory_tiling](experiments/07_shared_memory_tiling) | Use shared memory to improve locality |
| [08_streams_overlap](experiments/08_streams_overlap) | Explore CUDA streams and transfer/compute overlap |
| [08b_pinned_memory_overlap](experiments/08b_pinned_memory_overlap) | Compare pageable and pinned-memory stream behavior |
| 09_cuda_reduction | Compare reduction strategies |
| 10_nsight_profile_analysis | Profile CUDA workloads and summarize bottlenecks |

## Key Question

When does GPU acceleration help, and when does CPU/GPU coordination overhead dominate?

