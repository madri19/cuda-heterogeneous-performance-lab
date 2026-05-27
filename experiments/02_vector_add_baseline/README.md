# Experiment 02 - Vector Add Baseline

First CUDA kernel experiment.

## Goal

Verify the full CUDA execution path:

    host allocation
    -> device allocation
    -> host-to-device copy
    -> kernel launch
    -> device-to-host copy
    -> validation

## Kernel

Each CUDA thread computes one output element:

    c[i] = a[i] + b[i]

## Configuration

- Elements: 16,777,216
- Threads per block: 256
- Blocks: 65,536
- Data type: float

## Results

Initial cold run:

    Kernel elapsed: 153.483 ms
    Estimated kernel bandwidth: 1.22 GiB/sec
    Validation errors: 0

Warm runs:

    Kernel elapsed: 2.256 ms
    Estimated kernel bandwidth: 83.10 GiB/sec

    Kernel elapsed: 2.200 ms
    Estimated kernel bandwidth: 85.22 GiB/sec

    Kernel elapsed: 2.364 ms
    Estimated kernel bandwidth: 79.31 GiB/sec

## Interpretation

The first CUDA run was much slower than later runs.

This is expected because the first run can include one-time CUDA initialization, WSL/runtime warmup, GPU power-state transition, and other lazy setup costs.

The warm runs are the meaningful baseline for kernel timing.

## Key Lesson

Do not trust a single first-run CUDA timing.

CUDA experiments should use:

- warmup launches
- repeated timed iterations
- separate kernel timing from memory-transfer timing
- validation after execution

## Status

The CUDA kernel executed correctly and produced zero validation errors.

