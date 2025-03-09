#include <iostream>
#include <chrono>
#include <math.h>
#include <random>  

// function to perform intense computation on two arrays
void kernel(int n, float* x, float* y, float* output)
{
#pragma omp parallel for
//#pragma loop(no_vector)
    for (int i = 0; i < n; i++) {
        // Perform multiple floating-point operations on each element:
        float s = sinf(x[i]);                          // Sine of x[i]
        float c = cosf(y[i]);                          // Cosine of y[i]
        float m = s * c;                               // Multiply them
        float sq = sqrtf(fabsf(x[i] * y[i]) + 1.0f);   // Add 1.0f to avoid sqrt(0)

        // Combine everything with some arbitrary multiplications/additions:
        output[i] = (m + sq) * 1.2345f + y[i] * 0.9876f;
    }
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

   float* x      = new float[numElements];
   float* y      = new float[numElements];
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
   kernel(numElements, x, y, output);  

   // Stop time measurement and print the elapsed time.
   auto end = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double, std::milli> elapsed = end - start;  
   std::cout << "Kernel execution took: "  
       << elapsed.count() << " ms\n";  

   // Verify that the kernel function actually computes the correct results by
   // performing the same operations here and comparing the results with the
   // kernel output
   float maxError = 0.0f;
   for (int i = 0; i < numElements; i++) {
       float s = sinf(x[i]);                         // Sine of x[i]
       float c = cosf(y[i]);                         // Cosine of y[i]
       float m = s * c;                              // Multiply them
       float sq = sqrtf(fabsf(x[i] * y[i]) + 1.0f);  // Add 1.0f to avoid sqrt(0)

       // Combine everything with some arbitrary multiplications/additions:
       float expected = (m + sq) * 1.2345f + y[i] * 0.9876f;

       maxError = fmax(maxError, fabs(expected - output[i]));  
   }
   std::cout << "Max error: " << maxError << std::endl;

   // Free memory  
   delete[] x;
   delete[] y;
   delete[] output;

   return 0;  
}