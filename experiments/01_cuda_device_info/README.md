# Experiment 01 - CUDA Device Info

Captures CUDA runtime, driver, and device properties.

## Goal

Verify that CUDA is working in WSL and record the baseline GPU hardware properties used for later experiments.

## Hardware

- GPU: NVIDIA GeForce RTX 3060 Ti
- Compute capability: 8.6
- Global memory: 8191 MiB
- Multiprocessors: 38
- Warp size: 32
- Max threads per block: 1024
- Max threads per multiprocessor: 1536
- L2 cache: 3072 KiB
- Memory clock: 7001 MHz
- Memory bus width: 256 bits

## Environment

- Windows NVIDIA driver: 596.49
- CUDA runtime version: 13.2
- CUDA driver version: 13.2
- CUDA Toolkit: 13.2
- WSL Ubuntu 26.04

## Result

The CUDA runtime found one CUDA-capable GPU.

The device supports:

- unified addressing
- concurrent kernels
- host memory mapping

This confirms the local CUDA development environment is ready for Phase 3 experiments.

