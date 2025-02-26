#include <iostream>
#include <math.h>
#include <chrono>

#define MAX_EFFICIENCY 1

// function to add the elements of two arrays
__global__ void add(int numElements, float* x, float* y)
{
    unsigned int start_clock = clock();
    printf("Started thread: %d, blockid: %d, at time %d\n", threadIdx.x, blockIdx.x, start_clock);

#if MAX_EFFICIENCY == 1
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        y[i] = x[i] + y[i];
        printf("Valid index: %d, %d, %d, %d\n", threadIdx.x, blockIdx.x, blockDim.x, i);
    }
    else {
        printf("Invalid index ignored: %d\n", i);
    }

#else
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < numElements; i += stride)
        y[i] = x[i] + y[i];
#endif
}

int main(void)
{
    int N = 1 << 3; // Number of elements in 1-D vector

    float *x;
    float *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the CPU
    int blockSize = 32;
    int numBlocks = (N + blockSize - 1) / blockSize;

    std::cout << "Block Size: " << blockSize << "\n";
    std::cout << "Number of blocks: " << numBlocks << "\n";

    auto start = std::chrono::high_resolution_clock::now();
    add<<<numBlocks, blockSize>>>(N, x, y);
    
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Vector addition took: " << elapsed.count() << " milliseconds\n";

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}