#include <cuda_runtime.h>

#include <iostream>

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

int main() {
    int device_count = 0;

    CHECK_CUDA(cudaGetDeviceCount(&device_count));

    std::cout << "CUDA Device Info\n";
    std::cout << "Device count: " << device_count << "\n\n";

    int runtime_version = 0;
    int driver_version = 0;

    CHECK_CUDA(cudaRuntimeGetVersion(&runtime_version));
    CHECK_CUDA(cudaDriverGetVersion(&driver_version));

    std::cout << "CUDA runtime version: " << runtime_version << "\n";
    std::cout << "CUDA driver version:  " << driver_version << "\n\n";

    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp prop{};

        CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

        std::cout << "Device " << device << ": " << prop.name << "\n";
        std::cout << "Compute capability: "
                  << prop.major << "." << prop.minor << "\n";
        std::cout << "Global memory: "
                  << prop.totalGlobalMem / (1024 * 1024)
                  << " MiB\n";
        std::cout << "Shared memory per block: "
                  << prop.sharedMemPerBlock / 1024
                  << " KiB\n";
        std::cout << "Registers per block: "
                  << prop.regsPerBlock << "\n";
        std::cout << "Warp size: "
                  << prop.warpSize << "\n";
        std::cout << "Max threads per block: "
                  << prop.maxThreadsPerBlock << "\n";
        std::cout << "Max block dimensions: "
                  << prop.maxThreadsDim[0] << " x "
                  << prop.maxThreadsDim[1] << " x "
                  << prop.maxThreadsDim[2] << "\n";
        std::cout << "Max grid dimensions: "
                  << prop.maxGridSize[0] << " x "
                  << prop.maxGridSize[1] << " x "
                  << prop.maxGridSize[2] << "\n";
        std::cout << "Multiprocessor count: "
                  << prop.multiProcessorCount << "\n";
        std::cout << "Max threads per multiprocessor: "
                  << prop.maxThreadsPerMultiProcessor << "\n";
        int memory_clock_khz = 0;
        int memory_bus_width_bits = 0;

        CHECK_CUDA(cudaDeviceGetAttribute(
            &memory_clock_khz,
            cudaDevAttrMemoryClockRate,
            device
        ));

        CHECK_CUDA(cudaDeviceGetAttribute(
            &memory_bus_width_bits,
            cudaDevAttrGlobalMemoryBusWidth,
            device
        ));

        std::cout << "Memory clock rate: "
                  << memory_clock_khz / 1000
                  << " MHz\n";
        std::cout << "Memory bus width: "
                  << memory_bus_width_bits
                  << " bits\n";
        std::cout << "L2 cache size: "
                  << prop.l2CacheSize / 1024
                  << " KiB\n";
        std::cout << "Unified addressing: "
                  << (prop.unifiedAddressing ? "yes" : "no") << "\n";
        std::cout << "Concurrent kernels: "
                  << (prop.concurrentKernels ? "yes" : "no") << "\n";
        std::cout << "Can map host memory: "
                  << (prop.canMapHostMemory ? "yes" : "no") << "\n";
        std::cout << "\n";
    }

    return 0;
}
