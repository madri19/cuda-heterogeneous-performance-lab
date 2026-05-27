# Experiment 07 - Shared Memory Tiling

Compares naive matrix transpose against a shared-memory tiled transpose.

## Goal

Show how shared memory can improve global memory access patterns.

Matrix transpose is useful because the naive implementation creates inefficient global memory access on either the read or write side.

The tiled version stages data through shared memory to improve coalescing.

## Configuration

- Matrix: 8192 x 8192
- Elements: 67,108,864
- Bytes per matrix: 256 MiB
- Tile dimension: 32
- Block rows: 8
- Warmup iterations: 5
- Timed iterations: 30

## Results

| Kernel | Average Time | Estimated Bandwidth |
|---|---:|---:|
| Naive transpose | 4.0959 ms | 122.073 GiB/sec |
| Shared-memory tiled transpose | 1.32939 ms | 376.112 GiB/sec |

Tiled / naive speedup:

    3.08103x

Validation errors:

    0

## Interpretation

The shared-memory tiled transpose was about 3.08x faster than the naive transpose.

The tiled kernel improves memory behavior by loading a tile into shared memory, synchronizing the block, and writing the transposed tile back with better global-memory access patterns.

The extra shared-memory staging pays off because it reduces inefficient global-memory access.

## Key Lesson

Shared memory is useful when it changes the memory access pattern.

It is not automatically faster by itself. It helps when it enables coalesced global memory access, data reuse, or reduced global-memory traffic.

## Status

The tiled and naive transpose outputs matched with zero validation errors.

