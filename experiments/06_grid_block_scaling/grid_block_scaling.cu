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

constexpr int N = 1 << 26; // 67,108,864 floats
constexpr int WARMUP_ITERS = 5;
constexpr int TIMED_ITERS = 50;

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

int main() {
    std::cout << "CUDA Grid / Block Scaling Benchmark\n";
    std::cout << "Elements: " << N << "\n";
    std::cout << "Bytes per vector: "
              << (static_cast<size_t>(N) * sizeof(float)) / (1024 * 1024)
              << " MiB\n";
    std::cout << "Warmup iterations: " << WARMUP_ITERS << "\n";
    std::cout << "Timed iterations: " << TIMED_ITERS << "\n\n";

    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    std::vector<float> host_a(N);
    std::vector<float> host_b(N);
    std::vector<float> host_c(N);

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

    CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

    std::vector<int> block_sizes = {
        32,
        64,
        128,
        256,
        512,
        1024
    };

    double best_bandwidth = 0.0;
    int best_block_size = 0;

    for (int threads_per_block : block_sizes) {
        int blocks = (N + threads_per_block - 1) / threads_per_block;

        for (int iter = 0; iter < WARMUP_ITERS; ++iter) {
            vector_add_kernel<<<blocks, threads_per_block>>>(
                device_a,
                device_b,
                device_c,
                N
            );

            CHECK_CUDA(cudaGetLastError());
        }

        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start;
        cudaEvent_t stop;

        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));

        for (int iter = 0; iter < TIMED_ITERS; ++iter) {
            vector_add_kernel<<<blocks, threads_per_block>>>(
                device_a,
                device_b,
                device_c,
                N
            );

            CHECK_CUDA(cudaGetLastError());
        }

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

        double avg_ms = elapsed_ms / TIMED_ITERS;

        // Vector add reads a, reads b, writes c.
        double bytes_touched =
            static_cast<double>(N) * 3.0 * sizeof(float);

        double gib_per_second =
            bytes_touched / (avg_ms / 1000.0)
            / (1024.0 * 1024.0 * 1024.0);

        if (gib_per_second > best_bandwidth) {
            best_bandwidth = gib_per_second;
            best_block_size = threads_per_block;
        }

        std::cout << "Threads per block: "
                  << threads_per_block
                  << "\n";
        std::cout << "Blocks: "
                  << blocks
                  << "\n";
        std::cout << "Average kernel elapsed: "
                  << avg_ms
                  << " ms\n";
        std::cout << "Estimated bandwidth: "
                  << gib_per_second
                  << " GiB/sec\n";
        std::cout << "-----------------------------\n";

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    CHECK_CUDA(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;

    for (int i = 0; i < N; ++i) {
        float expected = host_a[i] + host_b[i];

        if (std::fabs(host_c[i] - expected) > 1e-5f) {
            ++errors;

            if (errors <= 10) {
                std::cerr << "Mismatch at " << i
                          << ": got " << host_c[i]
                          << ", expected " << expected
                          << "\n";
            }
        }
    }

    std::cout << "Best block size: "
              << best_block_size
              << "\n";
    std::cout << "Best estimated bandwidth: "
              << best_bandwidth
              << " GiB/sec\n";
    std::cout << "Validation errors: "
              << errors
              << "\n";

    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));

    return errors == 0 ? 0 : 1;
}
