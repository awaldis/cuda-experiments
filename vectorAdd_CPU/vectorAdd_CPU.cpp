#include <math.h>

#include <chrono>
#include <iostream>
#include <random>

// function to perform intense computation on two arrays
void Kernel(int n, float* x, float* y, float* output) {
#pragma omp parallel for
  // #pragma loop(no_vector)
  for (int i = 0; i < n; i++) {
    // Perform multiple floating-point operations on each element:
    float sine = sinf(x[i]);
    float cosine = cosf(y[i]);
    float mult_result = sine * cosine;

    // Add 1.0f to avoid sqrt(0)
    float square_root = sqrtf(fabsf(x[i] * y[i]) + 1.0F);

    // Combine everything with some arbitrary multiplications/additions:
    output[i] = (mult_result + square_root) * 1.2345F + y[i] * 0.9876F;
  }
}
//---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  // Use this default value if the user doesn't specify it.
  const int default_power_of_two = 20;
  int numElements = 1 << default_power_of_two;

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
  float* output = new float[numElements];

  // Generate random input data for the kernel to process.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(1.0, 2.0);

  for (int i = 0; i < numElements; i++) {
    x[i] = dis(gen);
    y[i] = dis(gen);
  }

  // Start time measurement.
  auto start = std::chrono::high_resolution_clock::now();

  // Run kernel code on the arrays in the CPU
  Kernel(numElements, x, y, output);

  // Stop time measurement and print the elapsed time.
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Kernel execution took: " << elapsed.count() << " ms\n";

  // Verify that the kernel function actually computes the correct results by
  // performing the same operations here and comparing the results with the
  // kernel output
  float maxError = 0.0F;
  for (int i = 0; i < numElements; i++) {
    float sine = sinf(x[i]);
    float cosine = cosf(y[i]);
    float mult_result = sine * cosine;

    // Add 1.0f to avoid sqrt(0)
    float square_root = sqrtf(fabsf(x[i] * y[i]) + 1.0F);

    // Combine everything with some arbitrary multiplications/additions:
    float expected = (mult_result + square_root) * 1.2345F + y[i] * 0.9876F;

    maxError = fmax(maxError, fabs(expected - output[i]));
  }
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  delete[] x;
  delete[] y;
  delete[] output;

  return 0;
}