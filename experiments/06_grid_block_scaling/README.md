# Experiment 06 - Grid / Block Scaling

Measures how CUDA block size affects vector-add kernel throughput.

## Goal

Explore how launch configuration affects performance.

The kernel computes:

    c[i] = a[i] + b[i]

Each configuration uses the same total problem size but changes the number of threads per block.

## Configuration

- Elements: 67,108,864
- Bytes per vector: 256 MiB
- Data type: float
- Warmup iterations: 5
- Timed iterations: 50

Tested block sizes:

- 32 threads/block
- 64 threads/block
- 128 threads/block
- 256 threads/block
- 512 threads/block
- 1024 threads/block

## Results

| Threads per Block | Blocks | Average Kernel Time | Estimated Bandwidth |
|---:|---:|---:|---:|
| 32 | 2,097,152 | 2.72138 ms | 275.595 GiB/sec |
| 64 | 1,048,576 | 1.92733 ms | 389.139 GiB/sec |
| 128 | 524,288 | 1.92219 ms | 390.18 GiB/sec |
| 256 | 262,144 | 1.92208 ms | 390.202 GiB/sec |
| 512 | 131,072 | 1.92262 ms | 390.092 GiB/sec |
| 1024 | 65,536 | 1.97241 ms | 380.246 GiB/sec |

Best observed block size:

    256 threads/block

Best estimated bandwidth:

    390.202 GiB/sec

Validation errors:

    0

## Interpretation

The 32-thread block configuration was significantly slower.

That configuration uses only one warp per block, which reduces scheduling flexibility and does not saturate the GPU as effectively.

Block sizes from 64 to 512 threads performed almost identically and reached about 390 GiB/sec.

The 1024-thread configuration was slightly slower than the best configurations.

## Key Lesson

Maximum block size is not automatically optimal.

For this memory-parallel vector-add workload, the stable high-performance range was 64 to 512 threads per block.

256 threads per block was the best observed configuration, but 128 and 512 were effectively equivalent.

## Status

All configurations produced correct results with zero validation errors.

