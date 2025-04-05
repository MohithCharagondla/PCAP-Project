#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

#define BLOCK_SIZE 16

// CUDA Kernel for Convolution-based Filters
__global__ void applyFilter(unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, float* d_kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half = kernelSize / 2;

    if (x >= half && y >= half && x < width - half && y < height - half) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    int idx = ((y + ky) * width + (x + kx)) * channels + c;
                    float weight = d_kernel[(ky + half) * kernelSize + (kx + half)];
                    sum += d_input[idx] * weight;
                }
            }
            int outIdx = (y * width + x) * channels + c;
            // Clamp values between 0 and 255
            d_output[outIdx] = min(max(int(sum), 0), 255);
        }
    } else if (x < width && y < height) {
        // Copy border pixels as is
        int idx = (y * width + x) * channels;
        for (int c = 0; c < channels; c++) {
            d_output[idx + c] = d_input[idx + c];
        }
    }
}

// CUDA Kernel for Color Inversion
__global__ void colorInversion(unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        for (int c = 0; c < channels; c++) {
            d_output[idx + c] = 255 - d_input[idx + c];
        }
    }
}

// CUDA Kernel for Black & White
__global__ void blackWhite(unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        // Use standard RGB to grayscale conversion weights
        unsigned char gray = 0.299f * d_input[idx] + 0.587f * d_input[idx + 1] + 0.114f * d_input[idx + 2];
        
        // Set all channels to the same gray value
        for (int c = 0; c < channels; c++) {
            d_output[idx + c] = gray;
        }
    }
}

// Function to generate filter kernels
void generateKernel(vector<float>& kernel, int kernelSize, int operation, int intensity) {
    // Adjust kernelSize based on intensity for blur
    int actualKernelSize = kernelSize;
    if (operation == 1) { // Blur
        // Larger kernel size for higher intensity
        actualKernelSize = 3 + (intensity - 1) * 2;  // More direct scaling with intensity
        actualKernelSize = min(actualKernelSize, 21); // Allow larger maximum kernel size
        
        // Create a box blur kernel
        kernel.resize(actualKernelSize * actualKernelSize);
        float value = 1.0f / (actualKernelSize * actualKernelSize);
        for (int i = 0; i < actualKernelSize * actualKernelSize; i++) {
            kernel[i] = value;
        }
    } else if (operation == 2) { // Sharpen
        // Use a 3x3 kernel for sharpen
        kernel.resize(3 * 3);
        
        // Make a more intense sharpen kernel while preserving brightness
        float centerValue = 1.0f + intensity;
        float surroundValue = -intensity / 4.0f; // Distribute negative values to maintain sum of 1
        
        // Create a sharpen kernel
        kernel = {
            0, surroundValue, 0,
            surroundValue, centerValue, surroundValue,
            0, surroundValue, 0
        };
        
        // Calculate current sum of kernel
        float kernelSum = centerValue + 4 * surroundValue;
        
        // Normalize to ensure kernel sum is 1
        if (abs(kernelSum - 1.0f) > 0.001f) {
            float normalizationFactor = 1.0f / kernelSum;
            for (int i = 0; i < kernel.size(); i++) {
                kernel[i] *= normalizationFactor;
            }
        }
    } else if (operation == 3) { // Outline
        // Use a 3x3 kernel for outline
        kernel.resize(3 * 3);
        
        // Create an outline/edge detection kernel (Laplacian)
        kernel = {
            -1, -1, -1,
            -1,  8, -1,
            -1, -1, -1
        };
    }
}

void processImage(Mat& inputImage, Mat& outputImage, int operation, int intensity) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();
    int imgSize = width * height * channels;
    
    // Allocate device memory
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);
    
    // Copy input image to device
    cudaMemcpy(d_input, inputImage.data, imgSize, cudaMemcpyHostToDevice);
    
    // Set up grid and block dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (operation >= 1 && operation <= 3) { // Convolution-based filters
        int baseKernelSize = 3;
        vector<float> kernel;
        generateKernel(kernel, baseKernelSize, operation, intensity);
        
        int actualKernelSize = sqrt(kernel.size());
        
        // Allocate and copy kernel to device
        float *d_kernel;
        cudaMalloc(&d_kernel, kernel.size() * sizeof(float));
        cudaMemcpy(d_kernel, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        // Apply filter
        applyFilter<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels, d_kernel, actualKernelSize);
        
        // Free kernel memory
        cudaFree(d_kernel);
    } else if (operation == 4) { // Color Inversion
        colorInversion<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    } else if (operation == 5) { // Black & White
        blackWhite<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    } else {
        // Invalid operation, just copy the input
        cudaMemcpy(d_output, d_input, imgSize, cudaMemcpyDeviceToDevice);
    }
    
    // Copy result back to host
    cudaMemcpy(outputImage.data, d_output, imgSize, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cout << "CUDA error: " << cudaGetErrorString(error) << endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cout << "Usage: image_filter.exe <image_path> <operation> <intensity>" << endl;
        cout << "Operations: 1=Blur, 2=Sharpen, 3=Outline, 4=Color Inversion, 5=Black & White" << endl;
        return -1;
    }

    string imagePath = argv[1];
    int operation = stoi(argv[2]);
    int intensity = stoi(argv[3]);
    
    // Clamp intensity to valid range
    intensity = max(1, min(intensity, 10));

    // Load image
    Mat image = imread(imagePath, IMREAD_COLOR);
    if (image.empty()) {
        cout << "Error: Could not open image at " << imagePath << endl;
        return -1;
    }

    Mat outputImage = Mat::zeros(image.size(), image.type());
    
    // Process the image
    processImage(image, outputImage, operation, intensity);
    
    // Save the output image
    imwrite("output.png", outputImage);
    cout << "Processed image saved as output.png" << endl;

    return 0;
}