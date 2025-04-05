#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <numeric> 

using namespace cv;
using namespace std;
using namespace std::chrono;

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
            d_output[outIdx] = min(max(int(sum), 0), 255);
        }
    } else if (x < width && y < height) {
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
        unsigned char gray = 0.299f * d_input[idx] + 0.587f * d_input[idx + 1] + 0.114f * d_input[idx + 2];
        for (int c = 0; c < channels; c++) {
            d_output[idx + c] = gray;
        }
    }
}

void generateKernel(vector<float>& kernel, int kernelSize, int operation, int intensity) {
    if (operation == 1) {
        int actualKernelSize = 3 + (intensity - 1) * 2;
        actualKernelSize = min(actualKernelSize, 21);
        kernel.resize(actualKernelSize * actualKernelSize);
        fill(kernel.begin(), kernel.end(), 1.0f / (actualKernelSize * actualKernelSize));
    } else if (operation == 2) {
        kernel = {0.0f, 
            -static_cast<float>(intensity)/4.0f, 
            0.0f,
            -static_cast<float>(intensity)/4.0f, 
            1.0f + static_cast<float>(intensity),
            -static_cast<float>(intensity)/4.0f,
            0.0f, 
            -static_cast<float>(intensity)/4.0f, 
            0.0f};
        float sum = accumulate(kernel.begin(), kernel.end(), 0.0f);
        for (float& val : kernel) val /= sum;
    } else if (operation == 3) {
        kernel = {-1, -1, -1,
                  -1, 8, -1,
                  -1, -1, -1};
    }
}

float processImage(Mat& inputImage, Mat& outputImage, int operation, int intensity) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();
    int imgSize = width * height * channels;

    // Initialize output image
    outputImage.create(height, width, inputImage.type());

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemcpy(d_input, inputImage.data, imgSize, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (operation >= 1 && operation <= 3) {
        vector<float> kernel;
        generateKernel(kernel, 3, operation, intensity);
        int actualKernelSize = sqrt(kernel.size());
        float *d_kernel;
        cudaMalloc(&d_kernel, kernel.size() * sizeof(float));
        cudaMemcpy(d_kernel, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice);
        applyFilter<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels, d_kernel, actualKernelSize);
        cudaFree(d_kernel);
    } else if (operation == 4) {
        colorInversion<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    } else if (operation == 5) {
        blackWhite<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    }

    cudaMemcpy(outputImage.data, d_output, imgSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cerr << "CUDA error: " << cudaGetErrorString(error) << endl;
    }

    return milliseconds;
}

void processImageCPU(Mat& inputImage, Mat& outputImage, int operation, int intensity) {
    outputImage = inputImage.clone();
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();

    if (operation >= 1 && operation <= 3) {
        vector<float> kernel;
        generateKernel(kernel, 3, operation, intensity);
        int kernelSize = sqrt(kernel.size());
        int half = kernelSize / 2;

        for (int y = half; y < height - half; ++y) {
            for (int x = half; x < width - half; ++x) {
                for (int c = 0; c < channels; ++c) {
                    float sum = 0.0f;
                    for (int ky = -half; ky <= half; ++ky) {
                        for (int kx = -half; kx <= half; ++kx) {
                            int imgX = x + kx;
                            int imgY = y + ky;
                            int idx = (imgY * width + imgX) * channels + c;
                            sum += inputImage.data[idx] * kernel[(ky + half) * kernelSize + (kx + half)];
                        }
                    }
                    int outIdx = (y * width + x) * channels + c;
                    outputImage.data[outIdx] = min(max(static_cast<int>(sum), 0), 255);
                }
            }
        }
    } else if (operation == 4) {
        for (int i = 0; i < width * height * channels; ++i) {
            outputImage.data[i] = 255 - inputImage.data[i];
        }
    } else if (operation == 5) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = (y * width + x) * channels;
                unsigned char gray = 0.299f * inputImage.data[idx] + 0.587f * inputImage.data[idx + 1] + 0.114f * inputImage.data[idx + 2];
                outputImage.data[idx] = gray;
                outputImage.data[idx + 1] = gray;
                outputImage.data[idx + 2] = gray;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <image_path> <operation> <intensity>" << endl;
        return -1;
    }

    string imagePath = argv[1];
    int operation = stoi(argv[2]);
    int intensity = stoi(argv[3]);
    intensity = max(1, min(intensity, 10));

    Mat image = imread(imagePath, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error loading image: " << imagePath << endl;
        return -1;
    }

    Mat outputImage, outputImageCPU;
    float cudaTime = processImage(image, outputImage, operation, intensity);
    imwrite("output.png", outputImage);

    auto start = high_resolution_clock::now();
    processImageCPU(image, outputImageCPU, operation, intensity);
    auto end = high_resolution_clock::now();
    float cpuTime = duration_cast<milliseconds>(end - start).count();
    imwrite("output_cpu.png", outputImageCPU);

    ofstream outFile("timing_results.txt", ios::app);
    if (outFile.is_open()) {
        outFile << "Operation: " << operation
                << ", CUDA Time: " << cudaTime << " ms, CPU Time: " << cpuTime << " ms" << endl;
        outFile.close();
    } else {
        cerr << "Failed to open timing file." << endl;
    }

    cout << "Processing completed. Timing results saved." << endl;
    return 0;
}