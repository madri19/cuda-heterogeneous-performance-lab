# Experiment 03 - CPU vs GPU Vector Add

Compares CPU vector addition against GPU kernel-only vector addition.

## Goal

Measure the raw compute/memory throughput difference between:

- CPU vector add loop
- CUDA vector add kernel

This experiment intentionally times only the GPU kernel after input data is already resident on the device.

Host/device transfer overhead is measured separately in Experiment 04.

## Configuration

- Elements: 16,777,216
- Data type: float
- Threads per block: 256
- Blocks: 65,536
- GPU warmup iterations: 5
- GPU timed iterations: 20
- CPU timed iterations: 20

## Results

| Target | Average Time | Estimated Bandwidth |
|---|---:|---:|
| CPU | 8.84773 ms | 21.1919 GiB/sec |
| GPU kernel | 0.483382 ms | 387.892 GiB/sec |

GPU kernel-only speedup:

    18.3038x

Validation errors:

    0

## Interpretation

The GPU kernel was much faster than the CPU loop for this large data-parallel vector operation.

This is the expected shape for a simple memory-parallel workload once data is already resident on the GPU.

The result does not yet include host/device transfer overhead.

## Key Lesson

GPU acceleration looks strong when measuring kernel-only throughput on a large parallel workload.

However, this does not prove end-to-end acceleration. For real application decisions, transfer cost and launch overhead must also be measured.

## Status

CPU and GPU results matched with zero validation errors.

