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
constexpr int WARMUP_ITERS = 5;
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
    std::cout << "Memory Transfer Overhead Benchmark\n";
    std::cout << "Elements: " << N << "\n";
    std::cout << "Threads per block: " << THREADS_PER_BLOCK << "\n";
    std::cout << "Warmup iterations: " << WARMUP_ITERS << "\n";
    std::cout << "Timed iterations: " << TIMED_ITERS << "\n\n";

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

    for (int iter = 0; iter < TIMED_ITERS; ++iter) {
        cpu_vector_add(host_a, host_b, cpu_c);
    }

    auto cpu_end = Clock::now();

    double cpu_total_ms =
        std::chrono::duration<double, std::milli>(
            cpu_end - cpu_start
        ).count();

    double cpu_avg_ms = cpu_total_ms / TIMED_ITERS;

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;

    CHECK_CUDA(cudaMalloc(&device_a, bytes));
    CHECK_CUDA(cudaMalloc(&device_b, bytes));
    CHECK_CUDA(cudaMalloc(&device_c, bytes));

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Warm up CUDA runtime, memory path, and kernel path.
    CHECK_CUDA(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

    for (int iter = 0; iter < WARMUP_ITERS; ++iter) {
        vector_add_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            device_a,
            device_b,
            device_c,
            N
        );

        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(gpu_c.data(), device_c, bytes, cudaMemcpyDeviceToHost));

    cudaEvent_t h2d_start;
    cudaEvent_t h2d_stop;
    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;
    cudaEvent_t d2h_start;
    cudaEvent_t d2h_stop;
    cudaEvent_t total_start;
    cudaEvent_t total_stop;

    CHECK_CUDA(cudaEventCreate(&h2d_start));
    CHECK_CUDA(cudaEventCreate(&h2d_stop));
    CHECK_CUDA(cudaEventCreate(&kernel_start));
    CHECK_CUDA(cudaEventCreate(&kernel_stop));
    CHECK_CUDA(cudaEventCreate(&d2h_start));
    CHECK_CUDA(cudaEventCreate(&d2h_stop));
    CHECK_CUDA(cudaEventCreate(&total_start));
    CHECK_CUDA(cudaEventCreate(&total_stop));

    float total_h2d_ms = 0.0f;
    float total_kernel_ms = 0.0f;
    float total_d2h_ms = 0.0f;
    float total_end_to_end_ms = 0.0f;

    for (int iter = 0; iter < TIMED_ITERS; ++iter) {
        CHECK_CUDA(cudaEventRecord(total_start));

        CHECK_CUDA(cudaEventRecord(h2d_start));

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

        CHECK_CUDA(cudaEventRecord(h2d_stop));
        CHECK_CUDA(cudaEventSynchronize(h2d_stop));

        CHECK_CUDA(cudaEventRecord(kernel_start));

        vector_add_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            device_a,
            device_b,
            device_c,
            N
        );

        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaEventRecord(kernel_stop));
        CHECK_CUDA(cudaEventSynchronize(kernel_stop));

        CHECK_CUDA(cudaEventRecord(d2h_start));

        CHECK_CUDA(cudaMemcpy(
            gpu_c.data(),
            device_c,
            bytes,
            cudaMemcpyDeviceToHost
        ));

        CHECK_CUDA(cudaEventRecord(d2h_stop));
        CHECK_CUDA(cudaEventSynchronize(d2h_stop));

        CHECK_CUDA(cudaEventRecord(total_stop));
        CHECK_CUDA(cudaEventSynchronize(total_stop));

        float h2d_ms = 0.0f;
        float kernel_ms = 0.0f;
        float d2h_ms = 0.0f;
        float end_to_end_ms = 0.0f;

        CHECK_CUDA(cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop));
        CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
        CHECK_CUDA(cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop));
        CHECK_CUDA(cudaEventElapsedTime(&end_to_end_ms, total_start, total_stop));

        total_h2d_ms += h2d_ms;
        total_kernel_ms += kernel_ms;
        total_d2h_ms += d2h_ms;
        total_end_to_end_ms += end_to_end_ms;
    }

    double avg_h2d_ms = total_h2d_ms / TIMED_ITERS;
    double avg_kernel_ms = total_kernel_ms / TIMED_ITERS;
    double avg_d2h_ms = total_d2h_ms / TIMED_ITERS;
    double avg_end_to_end_ms = total_end_to_end_ms / TIMED_ITERS;

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

    double vector_bytes = static_cast<double>(bytes);
    double kernel_bytes = static_cast<double>(N) * 3.0 * sizeof(float);
    double h2d_bytes = static_cast<double>(N) * 2.0 * sizeof(float);
    double d2h_bytes = vector_bytes;

    double h2d_gib =
        h2d_bytes / (avg_h2d_ms / 1000.0)
        / (1024.0 * 1024.0 * 1024.0);

    double d2h_gib =
        d2h_bytes / (avg_d2h_ms / 1000.0)
        / (1024.0 * 1024.0 * 1024.0);

    double kernel_gib =
        kernel_bytes / (avg_kernel_ms / 1000.0)
        / (1024.0 * 1024.0 * 1024.0);

    std::cout << "Blocks: " << blocks << "\n\n";

    std::cout << "CPU average elapsed: "
              << cpu_avg_ms
              << " ms\n\n";

    std::cout << "GPU H2D average: "
              << avg_h2d_ms
              << " ms\n";

    std::cout << "GPU kernel average: "
              << avg_kernel_ms
              << " ms\n";

    std::cout << "GPU D2H average: "
              << avg_d2h_ms
              << " ms\n";

    std::cout << "GPU end-to-end average: "
              << avg_end_to_end_ms
              << " ms\n\n";

    std::cout << "H2D bandwidth: "
              << h2d_gib
              << " GiB/sec\n";

    std::cout << "D2H bandwidth: "
              << d2h_gib
              << " GiB/sec\n";

    std::cout << "Kernel estimated bandwidth: "
              << kernel_gib
              << " GiB/sec\n\n";

    std::cout << "Kernel-only GPU / CPU speedup: "
              << cpu_avg_ms / avg_kernel_ms
              << "x\n";

    std::cout << "End-to-end GPU / CPU speedup: "
              << cpu_avg_ms / avg_end_to_end_ms
              << "x\n";

    std::cout << "Transfer share of GPU time: "
              << ((avg_h2d_ms + avg_d2h_ms) / avg_end_to_end_ms) * 100.0
              << "%\n";

    std::cout << "Validation errors: "
              << errors
              << "\n";

    CHECK_CUDA(cudaEventDestroy(h2d_start));
    CHECK_CUDA(cudaEventDestroy(h2d_stop));
    CHECK_CUDA(cudaEventDestroy(kernel_start));
    CHECK_CUDA(cudaEventDestroy(kernel_stop));
    CHECK_CUDA(cudaEventDestroy(d2h_start));
    CHECK_CUDA(cudaEventDestroy(d2h_stop));
    CHECK_CUDA(cudaEventDestroy(total_start));
    CHECK_CUDA(cudaEventDestroy(total_stop));

    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));

    return errors == 0 ? 0 : 1;
}
