#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "stb_image.h"
#include "stb_image_write.h"


// Gamma correction kernel function
__global__ void image_gamma(uint8_t *d_image, float gamma, int num_values) {
    int global_size = blockDim.x * gridDim.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (; idx < num_values; idx += global_size) {
        float value = d_image[idx] / 255.0f;
        value = powf(value, gamma);
        d_image[idx] = (uint8_t)(value * 255.0f);
    }
}

// Main function
int main(int argc, char** argv)
{
    int width, height, channels;

    // Load image
    uint8_t *h_image = stbi_load("/home/sabila/Project/CSCI642-Final-Project/image.jpg", &width, &height, &channels, 0);

    // Allocate memory for image
    int num_values = width * height * channels;
    uint8_t *d_image;
    hipMalloc((void**)&d_image, num_values * sizeof(uint8_t));

    float gpu_elapsed_time_ms;

    // Some events to count the execution time
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Copy image to device
    hipMemcpy(d_image, h_image, num_values * sizeof(uint8_t), hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (num_values + blockSize - 1) / blockSize;

    float gamma = 4.0;

    // Start to count execution time of GPU version
    hipEventRecord(start, 0);

    // Launch kernel
    image_gamma<<<gridSize, blockSize>>>(d_image, gamma, num_values);

    // Stop to count execution time of GPU version
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);

    // Copy image back to host
    hipMemcpy(h_image, d_image, num_values * sizeof(uint8_t), hipMemcpyDeviceToHost);

    // Calculate elapsed time
    hipEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("GPU time: %f ms\n", gpu_elapsed_time_ms);

    // Save image
    stbi_write_jpg("image_gamma.jpg", width, height, channels, h_image, 100);

    // Free memory
    hipFree(d_image);
    stbi_image_free(h_image);

    return 0;
}
