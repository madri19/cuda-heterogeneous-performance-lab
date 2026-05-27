#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA error: "                       \
                      << cudaGetErrorString(err)              \
                      << " at " << __FILE__                   \
                      << ":" << __LINE__ << "\n";             \
            return 1;                                         \
        }                                                     \
    } while (0)

constexpr int N = 1 << 24;
constexpr int CHUNKS = 8;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARMUP_ITERS = 3;
constexpr int TIMED_ITERS = 20;

__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

float run_sequential(
    const float* host_a,
    const float* host_b,
    float* host_c,
    float* device_a,
    float* device_b,
    float* device_c,
    int n,
    size_t bytes
) {
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    CHECK_CUDA(cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice));

    vector_add_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        device_a,
        device_b,
        device_c,
        n
    );

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(host_c, device_c, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return elapsed_ms;
}

float run_streamed(
    const float* host_a,
    const float* host_b,
    float* host_c,
    float* device_a,
    float* device_b,
    float* device_c,
    int n,
    int chunks
) {
    const int chunk_elements = n / chunks;
    const size_t chunk_bytes =
        static_cast<size_t>(chunk_elements) * sizeof(float);

    std::vector<cudaStream_t> streams(chunks);

    for (int i = 0; i < chunks; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int chunk = 0; chunk < chunks; ++chunk) {
        int offset = chunk * chunk_elements;

        int blocks =
            (chunk_elements + THREADS_PER_BLOCK - 1)
            / THREADS_PER_BLOCK;

        CHECK_CUDA(cudaMemcpyAsync(
            device_a + offset,
            host_a + offset,
            chunk_bytes,
            cudaMemcpyHostToDevice,
            streams[chunk]
        ));

        CHECK_CUDA(cudaMemcpyAsync(
            device_b + offset,
            host_b + offset,
            chunk_bytes,
            cudaMemcpyHostToDevice,
            streams[chunk]
        ));

        vector_add_kernel<<<blocks, THREADS_PER_BLOCK, 0, streams[chunk]>>>(
            device_a + offset,
            device_b + offset,
            device_c + offset,
            chunk_elements
        );

        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaMemcpyAsync(
            host_c + offset,
            device_c + offset,
            chunk_bytes,
            cudaMemcpyDeviceToHost,
            streams[chunk]
        ));
    }

    for (int i = 0; i < chunks; ++i) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    for (int i = 0; i < chunks; ++i) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return elapsed_ms;
}

int main() {
    std::cout << "CUDA Pinned Memory Stream Overlap Benchmark\n";
    std::cout << "Elements: " << N << "\n";
    std::cout << "Chunks: " << CHUNKS << "\n";
    std::cout << "Threads per block: " << THREADS_PER_BLOCK << "\n";
    std::cout << "Warmup iterations: " << WARMUP_ITERS << "\n";
    std::cout << "Timed iterations: " << TIMED_ITERS << "\n\n";

    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    float* host_a = nullptr;
    float* host_b = nullptr;
    float* host_seq_c = nullptr;
    float* host_stream_c = nullptr;

    CHECK_CUDA(cudaMallocHost(&host_a, bytes));
    CHECK_CUDA(cudaMallocHost(&host_b, bytes));
    CHECK_CUDA(cudaMallocHost(&host_seq_c, bytes));
    CHECK_CUDA(cudaMallocHost(&host_stream_c, bytes));

    for (int i = 0; i < N; ++i) {
        host_a[i] = static_cast<float>(i) * 0.5f;
        host_b[i] = static_cast<float>(i) * 2.0f;
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;

    CHECK_CUDA(cudaMalloc(&device_a, bytes));
    CHECK_CUDA(cudaMalloc(&device_b, bytes));
    CHECK_CUDA(cudaMalloc(&device_c, bytes));

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        run_sequential(
            host_a,
            host_b,
            host_seq_c,
            device_a,
            device_b,
            device_c,
            N,
            bytes
        );
    }

    float seq_total_ms = 0.0f;
    float stream_total_ms = 0.0f;

    for (int i = 0; i < TIMED_ITERS; ++i) {
        seq_total_ms += run_sequential(
            host_a,
            host_b,
            host_seq_c,
            device_a,
            device_b,
            device_c,
            N,
            bytes
        );
    }

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        run_streamed(
            host_a,
            host_b,
            host_stream_c,
            device_a,
            device_b,
            device_c,
            N,
            CHUNKS
        );
    }

    for (int i = 0; i < TIMED_ITERS; ++i) {
        stream_total_ms += run_streamed(
            host_a,
            host_b,
            host_stream_c,
            device_a,
            device_b,
            device_c,
            N,
            CHUNKS
        );
    }

    double seq_avg_ms = seq_total_ms / TIMED_ITERS;
    double stream_avg_ms = stream_total_ms / TIMED_ITERS;

    int errors = 0;

    for (int i = 0; i < N; ++i) {
        float expected = host_a[i] + host_b[i];

        if (std::fabs(host_seq_c[i] - expected) > 1e-5f ||
            std::fabs(host_stream_c[i] - expected) > 1e-5f) {
            ++errors;

            if (errors <= 10) {
                std::cerr << "Mismatch at " << i
                          << ": seq " << host_seq_c[i]
                          << ", stream " << host_stream_c[i]
                          << ", expected " << expected
                          << "\n";
            }
        }
    }

    std::cout << "Pinned sequential average: "
              << seq_avg_ms
              << " ms\n";

    std::cout << "Pinned streamed average:   "
              << stream_avg_ms
              << " ms\n";

    std::cout << "Pinned streamed / sequential speedup: "
              << seq_avg_ms / stream_avg_ms
              << "x\n";

    std::cout << "Validation errors: "
              << errors
              << "\n";

    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));

    CHECK_CUDA(cudaFreeHost(host_a));
    CHECK_CUDA(cudaFreeHost(host_b));
    CHECK_CUDA(cudaFreeHost(host_seq_c));
    CHECK_CUDA(cudaFreeHost(host_stream_c));

    return errors == 0 ? 0 : 1;
}
