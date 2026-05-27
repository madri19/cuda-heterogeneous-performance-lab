#include <cuda_runtime.h>

#include <chrono>
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

using Clock = std::chrono::steady_clock;

constexpr int N = 1 << 24;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int GPU_WARMUP_ITERS = 5;
constexpr int GPU_TIMED_ITERS = 20;
constexpr int CPU_TIMED_ITERS = 20;

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

void cpu_vector_add(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c
) {
    for (int i = 0; i < N; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    std::cout << "CPU vs GPU Vector Add\n";
    std::cout << "Elements: " << N << "\n";
    std::cout << "Threads per block: " << THREADS_PER_BLOCK << "\n";
    std::cout << "GPU warmup iterations: " << GPU_WARMUP_ITERS << "\n";
    std::cout << "GPU timed iterations: " << GPU_TIMED_ITERS << "\n";
    std::cout << "CPU timed iterations: " << CPU_TIMED_ITERS << "\n\n";

    size_t bytes = static_cast<size_t>(N) * sizeof(float);

    std::vector<float> host_a(N);
    std::vector<float> host_b(N);
    std::vector<float> cpu_c(N);
    std::vector<float> gpu_c(N);

    for (int i = 0; i < N; ++i) {
        host_a[i] = static_cast<float>(i) * 0.5f;
        host_b[i] = static_cast<float>(i) * 2.0f;
    }

    auto cpu_start = Clock::now();

    for (int iter = 0; iter < CPU_TIMED_ITERS; ++iter) {
        cpu_vector_add(host_a, host_b, cpu_c);
    }

    auto cpu_end = Clock::now();

    double cpu_elapsed_ms =
        std::chrono::duration<double, std::milli>(
            cpu_end - cpu_start
        ).count();

    double cpu_avg_ms = cpu_elapsed_ms / CPU_TIMED_ITERS;

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;

    CHECK_CUDA(cudaMalloc(&device_a, bytes));
    CHECK_CUDA(cudaMalloc(&device_b, bytes));
    CHECK_CUDA(cudaMalloc(&device_c, bytes));

    CHECK_CUDA(cudaMemcpy(
        device_a,
        host_a.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA(cudaMemcpy(
        device_b,
        host_b.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int iter = 0; iter < GPU_WARMUP_ITERS; ++iter) {
        vector_add_kernel<<<blocks, THREADS_PER_BLOCK>>>(
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

    for (int iter = 0; iter < GPU_TIMED_ITERS; ++iter) {
        vector_add_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            device_a,
            device_b,
            device_c,
            N
        );

        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float gpu_elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_elapsed_ms, start, stop));

    double gpu_avg_ms = gpu_elapsed_ms / GPU_TIMED_ITERS;

    CHECK_CUDA(cudaMemcpy(
        gpu_c.data(),
        device_c,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    int errors = 0;

    for (int i = 0; i < N; ++i) {
        if (std::fabs(gpu_c[i] - cpu_c[i]) > 1e-5f) {
            ++errors;

            if (errors <= 10) {
                std::cerr << "Mismatch at " << i
                          << ": gpu " << gpu_c[i]
                          << ", cpu " << cpu_c[i]
                          << "\n";
            }
        }
    }

    double bytes_touched = static_cast<double>(N) * 3.0 * sizeof(float);

    double cpu_bandwidth_gib =
        bytes_touched / (cpu_avg_ms / 1000.0)
        / (1024.0 * 1024.0 * 1024.0);

    double gpu_bandwidth_gib =
        bytes_touched / (gpu_avg_ms / 1000.0)
        / (1024.0 * 1024.0 * 1024.0);

    std::cout << "Blocks: " << blocks << "\n\n";

    std::cout << "CPU average elapsed: "
              << cpu_avg_ms
              << " ms\n";

    std::cout << "CPU estimated bandwidth: "
              << cpu_bandwidth_gib
              << " GiB/sec\n\n";

    std::cout << "GPU kernel average elapsed: "
              << gpu_avg_ms
              << " ms\n";

    std::cout << "GPU estimated kernel bandwidth: "
              << gpu_bandwidth_gib
              << " GiB/sec\n\n";

    std::cout << "GPU kernel / CPU speedup: "
              << cpu_avg_ms / gpu_avg_ms
              << "x\n";

    std::cout << "Validation errors: "
              << errors
              << "\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));

    return errors == 0 ? 0 : 1;
}
