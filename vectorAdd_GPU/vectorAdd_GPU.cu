#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// If set to 1, each thread does 1 element.
// Otherwise, threads stride through array.
#define MAX_EFFICIENCY 1

// Kernel function to add the elements of two arrays
__global__ void add(int numElements, float* x, float* y)
{
    // Print and record start time for this thread.
    unsigned int start_clock = clock();
    printf("Started thread: %d, blockid: %d, at time: %u\n",
        threadIdx.x, blockIdx.x, start_clock);

#if MAX_EFFICIENCY == 1
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        y[i] = x[i] + y[i];
        printf("Valid index: %d, %d, %d, %d\n",
            threadIdx.x, blockIdx.x, blockDim.x, i);
    }
    else {
        printf("Invalid index ignored: %d\n", i);
    }
#else
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < numElements; i += stride) {
        y[i] = x[i] + y[i];
    }
#endif

    // Print and record end time and execution time for this thread.
    unsigned int end_clock = clock();
    unsigned int clock_diff = end_clock - start_clock;
    printf("End thread: %d, blockid: %d, at time: %u, total time: %u\n",
        threadIdx.x, blockIdx.x, end_clock, clock_diff);
}

int main(int argc, char* argv[])
{
    // Use these default values if the user doesn't specify them.
    int blockSize = 32;
    int numBlocks = -1; // We'll compute if not given
    int numElements = 1 << 20;

    // -------------------------------------------------------
    // Parse command-line arguments (optional)
    // Usage: ./program <blockSize> <numBlocks> <numElements>
    // -------------------------------------------------------
    if (argc > 1) {
        blockSize = std::atoi(argv[1]);
    }
    if (argc > 2) {
        numBlocks = std::atoi(argv[2]);
    }
    if (argc > 3) {
        numElements = std::atoi(argv[3]);
    }

    // If numBlocks wasn't provided or was <= 0, compute it automatically
    if (numBlocks <= 0) {
        numBlocks = (numElements + blockSize - 1) / blockSize;
    }

    std::cout << "numElements : " << numElements << "\n";
    std::cout << "blockSize   : " << blockSize << "\n";
    std::cout << "numBlocks   : " << numBlocks << "\n";

    // Allocate Unified Memory – accessible from CPU or GPU
    float* x, * y;
    cudaMallocManaged(&x, numElements * sizeof(float));
    cudaMallocManaged(&y, numElements * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < numElements; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Measure time on the host
    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    add<<<numBlocks, blockSize>>>(numElements, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Kernel execution took: "
        << elapsed.count() << " ms\n";

    // Check for errors (all values should be 3.0f after y[i] += x[i])
    float maxError = 0.0f;
    for (int i = 0; i < numElements; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
