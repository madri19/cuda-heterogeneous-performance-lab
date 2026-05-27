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

constexpr int N = 1 << 26; // 67,108,864 floats = 256 MiB
constexpr int WARMUP_ITERS = 5;
constexpr int TIMED_ITERS = 50;

__global__ void copy_kernel(
    const float* src,
    float* dst,
    int n
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        dst[index] = src[index];
    }
}

int main() {
    std::cout << "CUDA Memory Bandwidth Benchmark\n";
    std::cout << "Elements: " << N << "\n";
    std::cout << "Bytes per buffer: "
              << (static_cast<size_t>(N) * sizeof(float)) / (1024 * 1024)
              << " MiB\n";
    std::cout << "Warmup iterations: " << WARMUP_ITERS << "\n";
    std::cout << "Timed iterations: " << TIMED_ITERS << "\n\n";

    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    std::vector<float> host_src(N);
    std::vector<float> host_dst(N);

    for (int i = 0; i < N; ++i) {
        host_src[i] = static_cast<float>(i) * 0.25f;
    }

    float* device_src = nullptr;
    float* device_dst = nullptr;

    CHECK_CUDA(cudaMalloc(&device_src, bytes));
    CHECK_CUDA(cudaMalloc(&device_dst, bytes));

    CHECK_CUDA(cudaMemcpy(
        device_src,
        host_src.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    std::vector<int> block_sizes = {
        128,
        256,
        512,
        1024
    };

    for (int threads_per_block : block_sizes) {
        int blocks = (N + threads_per_block - 1) / threads_per_block;

        for (int iter = 0; iter < WARMUP_ITERS; ++iter) {
            copy_kernel<<<blocks, threads_per_block>>>(
                device_src,
                device_dst,
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
            copy_kernel<<<blocks, threads_per_block>>>(
                device_src,
                device_dst,
                N
            );

            CHECK_CUDA(cudaGetLastError());
        }

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

        double avg_ms = elapsed_ms / TIMED_ITERS;

        // Copy kernel reads src and writes dst.
        double bytes_touched =
            static_cast<double>(bytes) * 2.0;

        double gib_per_second =
            bytes_touched / (avg_ms / 1000.0)
            / (1024.0 * 1024.0 * 1024.0);

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

    CHECK_CUDA(cudaMemcpy(
        host_dst.data(),
        device_dst,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    int errors = 0;

    for (int i = 0; i < N; ++i) {
        if (std::fabs(host_dst[i] - host_src[i]) > 1e-5f) {
            ++errors;

            if (errors <= 10) {
                std::cerr << "Mismatch at " << i
                          << ": got " << host_dst[i]
                          << ", expected " << host_src[i]
                          << "\n";
            }
        }
    }

    std::cout << "Validation errors: " << errors << "\n";

    CHECK_CUDA(cudaFree(device_src));
    CHECK_CUDA(cudaFree(device_dst));

    return errors == 0 ? 0 : 1;
}
