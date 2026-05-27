# Experiment 08 - CUDA Streams and Overlap

Compares sequential copy/kernel/copy execution against a chunked CUDA streams version.

## Goal

Test whether CUDA streams improve end-to-end vector-add time by overlapping:

- host-to-device copies
- kernel execution
- device-to-host copies

## Configuration

- Elements: 16,777,216
- Chunks: 8
- Threads per block: 256
- Warmup iterations: 3
- Timed iterations: 20
- Host memory: pageable `std::vector` memory

## Results

| Mode | Average Time |
|---|---:|
| Sequential | 19.9095 ms |
| Streamed | 20.6194 ms |

Streamed / sequential speedup:

    0.965574x

Validation errors:

    0

## Interpretation

The streamed version was slightly slower than the sequential version.

This means the naive use of CUDA streams did not produce useful overlap for this workload.

Likely causes:

- host memory was pageable, not pinned
- transfer operations may have serialized
- per-stream scheduling overhead offset any overlap
- the kernel was small relative to transfer time
- WSL/driver behavior may affect overlap visibility

## Key Lesson

CUDA streams do not automatically make a workload faster.

Streams are useful when the workload and memory setup allow real overlap. In practice this often requires pinned host memory, enough compute per chunk, and careful chunk sizing.

## Status

Sequential and streamed outputs matched with zero validation errors.

