# Experiment 04 - Memory Transfer Overhead

Measures CPU time, GPU kernel-only time, host/device transfer time, and full GPU end-to-end time for vector addition.

## Goal

Show the difference between GPU kernel speed and full end-to-end GPU execution time.

This experiment separates:

- host-to-device transfer
- CUDA kernel execution
- device-to-host transfer
- full GPU end-to-end time
- CPU execution time

## Configuration

- Elements: 16,777,216
- Data type: float
- Threads per block: 256
- Blocks: 65,536
- Warmup iterations: 5
- Timed iterations: 20

## Results

| Measurement | Average Time |
|---|---:|
| CPU | 9.32453 ms |
| GPU H2D | 12.7318 ms |
| GPU kernel | 0.509534 ms |
| GPU D2H | 6.8729 ms |
| GPU end-to-end | 20.2527 ms |

Bandwidth estimates:

| Path | Bandwidth |
|---|---:|
| H2D | 9.81795 GiB/sec |
| D2H | 9.09369 GiB/sec |
| GPU kernel | 367.983 GiB/sec |

Speedups:

    Kernel-only GPU / CPU speedup: 18.3001x
    End-to-end GPU / CPU speedup: 0.46041x

Transfer share of GPU time:

    96.8005%

Validation errors:

    0

## Interpretation

The GPU kernel was much faster than the CPU loop.

However, the full GPU path was slower than the CPU because host/device memory transfer dominated the runtime.

The GPU spent only about 0.51 ms in the kernel, but the end-to-end GPU path took about 20.25 ms.

Most GPU-path time came from transferring data across the host/device boundary.

## Key Lesson

Kernel-only timing can be misleading.

A GPU can be much faster at computation while still losing end-to-end if the workload requires large host-to-device and device-to-host transfers.

For GPU acceleration to pay off, one or more of these must be true:

- data already lives on the GPU
- many kernels reuse the same device-resident data
- computation per transferred byte is high
- transfers overlap with compute
- transfer volume is reduced

## Status

The GPU result matched the CPU result with zero validation errors.

