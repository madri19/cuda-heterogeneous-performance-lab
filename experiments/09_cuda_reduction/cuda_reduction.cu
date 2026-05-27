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

constexpr int N = 1 << 26; // 67,108,864 floats
constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARMUP_ITERS = 5;
constexpr int TIMED_ITERS = 30;

__global__ void reduce_sum_kernel(
    const float* input,
    float* block_sums,
    int n
) {
    __shared__ float shared[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;

    if (index < n) {
        sum += input[index];
    }

    if (index + blockDim.x < n) {
        sum += input[index + blockDim.x];
    }

    shared[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = shared[0];
    }
}

double cpu_sum(const std::vector<float>& data) {
    double sum = 0.0;

    for (float value : data) {
        sum += value;
    }

    return sum;
}

int main() {
    std::cout << "CUDA Reduction Benchmark\n";
    std::cout << "Elements: " << N << "\n";
    std::cout << "Threads per block: " << THREADS_PER_BLOCK << "\n";
    std::cout << "Warmup iterations: " << WARMUP_ITERS << "\n";
    std::cout << "Timed iterations: " << TIMED_ITERS << "\n\n";

    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    std::vector<float> host_input(N);

    for (int i = 0; i < N; ++i) {
        host_input[i] = 1.0f;
    }

    auto cpu_start = Clock::now();

    double cpu_result = 0.0;

    for (int iter = 0; iter < TIMED_ITERS; ++iter) {
        cpu_result = cpu_sum(host_input);
    }

    auto cpu_end = Clock::now();

    double cpu_total_ms =
        std::chrono::duration<double, std::milli>(
            cpu_end - cpu_start
        ).count();

    double cpu_avg_ms = cpu_total_ms / TIMED_ITERS;

    float* device_input = nullptr;
    float* device_block_sums = nullptr;

    int blocks =
        (N + THREADS_PER_BLOCK * 2 - 1)
        / (THREADS_PER_BLOCK * 2);

    std::vector<float> host_block_sums(blocks);

    CHECK_CUDA(cudaMalloc(&device_input, bytes));
    CHECK_CUDA(cudaMalloc(
        &device_block_sums,
        static_cast<size_t>(blocks) * sizeof(float)
    ));

    CHECK_CUDA(cudaMemcpy(
        device_input,
        host_input.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    for (int iter = 0; iter < WARMUP_ITERS; ++iter) {
        reduce_sum_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            device_input,
            device_block_sums,
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
        reduce_sum_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            device_input,
            device_block_sums,
            N
        );

        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float gpu_total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_total_ms, start, stop));

    double gpu_avg_ms = gpu_total_ms / TIMED_ITERS;

    CHECK_CUDA(cudaMemcpy(
        host_block_sums.data(),
        device_block_sums,
        static_cast<size_t>(blocks) * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    double gpu_result = 0.0;

    for (float partial : host_block_sums) {
        gpu_result += partial;
    }

    double bytes_read = static_cast<double>(bytes);

    double cpu_bandwidth =
        bytes_read / (cpu_avg_ms / 1000.0)
        / (1024.0 * 1024.0 * 1024.0);

    double gpu_bandwidth =
        bytes_read / (gpu_avg_ms / 1000.0)
        / (1024.0 * 1024.0 * 1024.0);

    double diff = std::fabs(cpu_result - gpu_result);

    std::cout << "Blocks: " << blocks << "\n\n";

    std::cout << "CPU average elapsed: "
              << cpu_avg_ms
              << " ms\n";

    std::cout << "CPU estimated read bandwidth: "
              << cpu_bandwidth
              << " GiB/sec\n";

    std::cout << "CPU result: "
              << cpu_result
              << "\n\n";

    std::cout << "GPU kernel average elapsed: "
              << gpu_avg_ms
              << " ms\n";

    std::cout << "GPU estimated read bandwidth: "
              << gpu_bandwidth
              << " GiB/sec\n";

    std::cout << "GPU result: "
              << gpu_result
              << "\n\n";

    std::cout << "GPU kernel / CPU speedup: "
              << cpu_avg_ms / gpu_avg_ms
              << "x\n";

    std::cout << "Absolute difference: "
              << diff
              << "\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_block_sums));

    return diff < 1e-3 ? 0 : 1;
}
