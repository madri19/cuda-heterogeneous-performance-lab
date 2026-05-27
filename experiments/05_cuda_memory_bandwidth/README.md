# Experiment 05 - CUDA Memory Bandwidth

Measures device-only global memory bandwidth using a CUDA copy kernel.

## Goal

Isolate GPU global memory bandwidth without host/device transfer overhead.

The kernel performs:

    dst[i] = src[i]

This touches two global memory streams:

- one read from `src`
- one write to `dst`

## Configuration

- Elements: 67,108,864
- Bytes per buffer: 256 MiB
- Data type: float
- Warmup iterations: 5
- Timed iterations: 50

Tested block sizes:

- 128 threads/block
- 256 threads/block
- 512 threads/block
- 1024 threads/block

## Results

| Threads per Block | Blocks | Average Kernel Time | Estimated Bandwidth |
|---:|---:|---:|---:|
| 128 | 524,288 | 1.29638 ms | 385.688 GiB/sec |
| 256 | 262,144 | 1.29757 ms | 385.335 GiB/sec |
| 512 | 131,072 | 1.31035 ms | 381.577 GiB/sec |
| 1024 | 65,536 | 1.39874 ms | 357.464 GiB/sec |

Validation errors:

    0

## Interpretation

The device-only memory copy kernel reached approximately 386 GiB/sec at the best observed block sizes.

Block sizes from 128 to 512 performed similarly.

The 1024-thread block configuration was slower, likely due to lower scheduling flexibility or occupancy/resource effects.

## Key Lesson

For this simple memory bandwidth workload, larger blocks were not automatically better.

The best observed configurations used 128 or 256 threads per block.

This reinforces the need to measure launch configuration choices instead of assuming maximum block size is optimal.

## Status

The device copy result matched the source data with zero validation errors.

