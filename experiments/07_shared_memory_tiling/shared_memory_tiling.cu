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

constexpr int WIDTH = 8192;
constexpr int HEIGHT = 8192;
constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;
constexpr int WARMUP_ITERS = 5;
constexpr int TIMED_ITERS = 30;

__global__ void transpose_naive(
    const float* input,
    float* output,
    int width,
    int height
) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            output[x * height + (y + j)] =
                input[(y + j) * width + x];
        }
    }
}

__global__ void transpose_tiled(
    const float* input,
    float* output,
    int width,
    int height
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] =
                input[(y + j) * width + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            output[(y + j) * height + x] =
                tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

float benchmark_kernel(
    void (*kernel)(const float*, float*, int, int),
    const float* device_input,
    float* device_output,
    dim3 grid,
    dim3 block
) {
    for (int iter = 0; iter < WARMUP_ITERS; ++iter) {
        kernel<<<grid, block>>>(
            device_input,
            device_output,
            WIDTH,
            HEIGHT
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
        kernel<<<grid, block>>>(
            device_input,
            device_output,
            WIDTH,
            HEIGHT
        );

        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return elapsed_ms / TIMED_ITERS;
}

int main() {
    std::cout << "Shared Memory Tiling Benchmark\n";
    std::cout << "Matrix: " << WIDTH << " x " << HEIGHT << "\n";
    std::cout << "Tile dim: " << TILE_DIM << "\n";
    std::cout << "Block rows: " << BLOCK_ROWS << "\n";
    std::cout << "Warmup iterations: " << WARMUP_ITERS << "\n";
    std::cout << "Timed iterations: " << TIMED_ITERS << "\n\n";

    const size_t elements =
        static_cast<size_t>(WIDTH) * static_cast<size_t>(HEIGHT);

    const size_t bytes = elements * sizeof(float);

    std::cout << "Elements: " << elements << "\n";
    std::cout << "Bytes per matrix: "
              << bytes / (1024 * 1024)
              << " MiB\n\n";

    std::vector<float> host_input(elements);
    std::vector<float> host_output_naive(elements);
    std::vector<float> host_output_tiled(elements);

    for (size_t i = 0; i < elements; ++i) {
        host_input[i] = static_cast<float>(i % 1024);
    }

    float* device_input = nullptr;
    float* device_output_naive = nullptr;
    float* device_output_tiled = nullptr;

    CHECK_CUDA(cudaMalloc(&device_input, bytes));
    CHECK_CUDA(cudaMalloc(&device_output_naive, bytes));
    CHECK_CUDA(cudaMalloc(&device_output_tiled, bytes));

    CHECK_CUDA(cudaMemcpy(
        device_input,
        host_input.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid(
        (WIDTH + TILE_DIM - 1) / TILE_DIM,
        (HEIGHT + TILE_DIM - 1) / TILE_DIM
    );

    float naive_ms = benchmark_kernel(
        transpose_naive,
        device_input,
        device_output_naive,
        grid,
        block
    );

    float tiled_ms = benchmark_kernel(
        transpose_tiled,
        device_input,
        device_output_tiled,
        grid,
        block
    );

    CHECK_CUDA(cudaMemcpy(
        host_output_naive.data(),
        device_output_naive,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    CHECK_CUDA(cudaMemcpy(
        host_output_tiled.data(),
        device_output_tiled,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    int errors = 0;

    for (size_t i = 0; i < elements; ++i) {
        if (std::fabs(host_output_naive[i] - host_output_tiled[i]) > 1e-5f) {
            ++errors;

            if (errors <= 10) {
                std::cerr << "Mismatch at " << i
                          << ": naive " << host_output_naive[i]
                          << ", tiled " << host_output_tiled[i]
                          << "\n";
            }
        }
    }

    // Matrix transpose reads one matrix and writes one matrix.
    double bytes_touched = static_cast<double>(bytes) * 2.0;

    double naive_bandwidth =
        bytes_touched / (naive_ms / 1000.0)
        / (1024.0 * 1024.0 * 1024.0);

    double tiled_bandwidth =
        bytes_touched / (tiled_ms / 1000.0)
        / (1024.0 * 1024.0 * 1024.0);

    std::cout << "Naive transpose average: "
              << naive_ms
              << " ms\n";
    std::cout << "Naive estimated bandwidth: "
              << naive_bandwidth
              << " GiB/sec\n\n";

    std::cout << "Tiled transpose average: "
              << tiled_ms
              << " ms\n";
    std::cout << "Tiled estimated bandwidth: "
              << tiled_bandwidth
              << " GiB/sec\n\n";

    std::cout << "Tiled / naive speedup: "
              << naive_ms / tiled_ms
              << "x\n";

    std::cout << "Validation errors: "
              << errors
              << "\n";

    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_output_naive));
    CHECK_CUDA(cudaFree(device_output_tiled));

    return errors == 0 ? 0 : 1;
}
