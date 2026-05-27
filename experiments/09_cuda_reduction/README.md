# Experiment 09 - CUDA Reduction

Compares CPU summation against a CUDA block-level reduction.

## Goal

Measure a reduction workload, which is different from vector add because many input elements collapse into fewer output values.

This experiment performs the first GPU reduction stage:

    input array -> per-block partial sums

The final sum of block partials is completed on the CPU.

## Configuration

- Elements: 67,108,864
- Data type: float
- Threads per block: 256
- Blocks: 131,072
- Warmup iterations: 5
- Timed iterations: 30
- Input values: all 1.0f

## Results

| Target | Average Time | Estimated Read Bandwidth | Result |
|---|---:|---:|---:|
| CPU | 44.1059 ms | 5.66817 GiB/sec | 6.71089e+07 |
| GPU kernel | 1.01836 ms | 245.492 GiB/sec | 6.71089e+07 |

GPU kernel / CPU speedup:

    43.3106x

Absolute difference:

    0

## Interpretation

The CUDA reduction kernel was much faster than the CPU summation loop.

The GPU reduction benefits from massive parallelism and shared-memory block-local accumulation.

This is a kernel-only measurement. Host/device transfer overhead is not included.

## Key Lesson

Reductions require a different GPU pattern than one-thread-per-output kernels.

A useful reduction kernel usually combines:

- coalesced global reads
- per-thread partial work
- shared-memory accumulation
- block-level partial sums
- a final reduction stage

This experiment implements the first reduction stage and validates the result against the CPU.

## Status

The GPU and CPU results matched exactly.

