#include <math.h>

#include <chrono>
#include <iostream>
#include <random>

const float kRandomDistributionMin = 1.0F;
const float kRandomDistributionMax = 2.0F;

// The actual values of these constants are not important.  As long
// as they are not something trivial like zero or one.
const float kMagicNumber1 = 1.2345F;
const float kMagicNumber2 = 0.9876F;

//---------------------------------------------------------------------------
// The purpose of this function is to perform intense computation on two
// arrays so that the caller can measure how long it takes to execute.
//---------------------------------------------------------------------------
void Kernel(int n, float* input_array_1, float* input_array_2, float* output) {
#pragma omp parallel for
  // #pragma loop(no_vector)
  for (int i = 0; i < n; i++) {
    // Perform multiple floating-point operations on each element:
    float sine = sinf(input_array_1[i]);
    float cosine = cosf(input_array_2[i]);
    float mult_result = sine * cosine;

    // Add 1.0f to avoid sqrt(0)
    float square_root =
        sqrtf(fabsf(input_array_1[i] * input_array_2[i]) + 1.0F);

    // Combine everything with some arbitrary multiplications/additions:
    output[i] = ((mult_result + square_root) * kMagicNumber1) +
                (input_array_2[i] * kMagicNumber2);
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

  float* input_array_1 = new float[numElements];
  float* input_array_2 = new float[numElements];
  float* output = new float[numElements];

  // Generate random input data for the kernel to process.
  std::random_device rand;
  std::mt19937 gen(rand());
  std::uniform_real_distribution<float> dis(kRandomDistributionMin,
                                            kRandomDistributionMax);

  for (int i = 0; i < numElements; i++) {
    input_array_1[i] = dis(gen);
    input_array_2[i] = dis(gen);
  }

  // Start time measurement.
  auto start = std::chrono::high_resolution_clock::now();

  // Run kernel code on the arrays in the CPU
  Kernel(numElements, input_array_1, input_array_2, output);

  // Stop time measurement and print the elapsed time.
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Kernel execution took: " << elapsed.count() << " ms\n";

  // Verify that the kernel function actually computes the correct results by
  // performing the same operations here and comparing the results with the
  // kernel output
  float maxError = 0.0F;
  for (int i = 0; i < numElements; i++) {
    float sine = sinf(input_array_1[i]);
    float cosine = cosf(input_array_2[i]);
    float mult_result = sine * cosine;

    // Add 1.0f to avoid sqrt(0)
    float square_root =
        sqrtf(fabsf(input_array_1[i] * input_array_2[i]) + 1.0F);

    // Combine everything with some arbitrary multiplications/additions:
    float expected = ((mult_result + square_root) * kMagicNumber1) +
                     (input_array_2[i] * kMagicNumber2);

    maxError = fmax(maxError, fabs(expected - output[i]));
  }
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  delete[] input_array_1;
  delete[] input_array_2;
  delete[] output;

  return 0;
}