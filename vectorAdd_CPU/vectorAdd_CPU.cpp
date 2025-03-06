#include <iostream>
#include <chrono>
#include <math.h>

// function to add the elements of two arrays
void add(int n, float* x, float* y)
{
   for (int i = 0; i < n; i++)
       y[i] = x[i] + y[i];
}

int main(int argc, char* argv[])
{
    // Use these default values if the user doesn't specify them.
    int numElements = 1 << 20;

    // -------------------------------------------------------
    // Parse command-line arguments (optional)
    // Usage: ./program <numElements>
    // -------------------------------------------------------
    if (argc > 1) {
        numElements = std::atoi(argv[1]);
    }

    std::cout << "numElements : " << numElements << "\n";

    float* x = new float[numElements];
    float* y = new float[numElements];

    // initialize x and y arrays on the host
    for (int i = 0; i < numElements; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Measure time on the host
    auto start = std::chrono::high_resolution_clock::now();

    // Run kernel on 1M elements on the CPU
    add(numElements, x, y);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Vector addition execution took: "
        << elapsed.count() << " ms\n";

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < numElements; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    delete[] x;
    delete[] y;

    return 0;
}