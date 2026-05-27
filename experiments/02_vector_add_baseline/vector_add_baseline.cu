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

constexpr int N = 1 << 24;          // 16,777,216 floats
constexpr int THREADS_PER_BLOCK = 256;

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
    std::cout << "Vector Add Baseline\n";
    std::cout << "Elements: " << N << "\n";
    std::cout << "Threads per block: " << THREADS_PER_BLOCK << "\n\n";

    size_t bytes = static_cast<size_t>(N) * sizeof(float);

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

    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    vector_add_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        device_a,
        device_b,
        device_c,
        N
    );

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CHECK_CUDA(cudaMemcpy(
        host_c.data(),
        device_c,
        bytes,
        cudaMemcpyDeviceToHost
    ));

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

    double seconds = elapsed_ms / 1000.0;
    double elements_per_second = N / seconds;

    // Per element: read a, read b, write c.
    double bytes_touched = static_cast<double>(N) * 3.0 * sizeof(float);
    double gib_per_second =
        bytes_touched / seconds / (1024.0 * 1024.0 * 1024.0);

    std::cout << "Blocks: " << blocks << "\n";
    std::cout << "Kernel elapsed: " << elapsed_ms << " ms\n";
    std::cout << "Throughput: " << elements_per_second << " elements/sec\n";
    std::cout << "Estimated kernel bandwidth: "
              << gib_per_second
              << " GiB/sec\n";
    std::cout << "Validation errors: " << errors << "\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(device_a));
    CHECK_CUDA(cudaFree(device_b));
    CHECK_CUDA(cudaFree(device_c));

    return errors == 0 ? 0 : 1;
}
