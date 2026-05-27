# Experiment 08B - Pinned Memory Stream Overlap

Compares sequential and streamed vector-add execution using pinned host memory.

## Goal

Follow up Experiment 08 by replacing pageable host memory with pinned host memory.

Pinned memory enables faster and more predictable host/device transfers and is usually required for effective asynchronous copy overlap.

## Configuration

- Elements: 16,777,216
- Chunks: 8
- Threads per block: 256
- Warmup iterations: 3
- Timed iterations: 20
- Host memory: pinned via `cudaMallocHost`

## Results

| Mode | Average Time |
|---|---:|
| Pinned sequential | 8.46601 ms |
| Pinned streamed | 8.73883 ms |

Pinned streamed / sequential speedup:

    0.968781x

Validation errors:

    0

## Comparison to Pageable Memory

Experiment 08 pageable sequential time:

    19.9095 ms

Experiment 08B pinned sequential time:

    8.46601 ms

Pinned host memory significantly reduced transfer-heavy end-to-end time.

## Interpretation

Pinned memory helped substantially.

Streams still did not improve performance for this workload. The streamed version was slightly slower than sequential execution.

Likely causes:

- the vector-add kernel is too small relative to transfer cost
- 8 chunks add scheduling overhead
- transfer/compute overlap is limited by workload shape
- WSL/driver behavior may limit visible overlap

## Key Lesson

Pinned memory is a major improvement for transfer-heavy CUDA workloads.

Streams are not automatically faster. They need enough independent work, good chunk sizing, and enough compute per transfer to hide copy cost.

## Status

Sequential and streamed outputs matched with zero validation errors.

