#include <iostream>
#include <windows.h>
#include <gdiplus.h>
#include <cuda_runtime.h>
#include <cmath>
#include <device_launch_parameters.h>
#pragma comment (lib, "gdiplus.lib")

using namespace Gdiplus;

// CUDA Kernel for MSE Calculation
__global__ void calculateMSE(const unsigned char* img1, const unsigned char* img2, float* mse, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float shared_mse[256];  // Shared memory for partial sums

    if (idx < size) {
        float diff = img1[idx] - img2[idx];
        shared_mse[threadIdx.x] = diff * diff;
    }
    else {
        shared_mse[threadIdx.x] = 0.0f;
    }

    __syncthreads();  // Synchronize threads within the block

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_mse[threadIdx.x] += shared_mse[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(mse, shared_mse[0]);  // Atomic addition of partial sums for MSE
    }
}

// CUDA Kernel for Cosine Similarity Calculation
__global__ void calculateCosine(const unsigned char* img1, const unsigned char* img2, float* dotProduct, float* norm1, float* norm2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float shared_dot[256];
    __shared__ float shared_norm1[256];
    __shared__ float shared_norm2[256];

    if (idx < size) {
        shared_dot[threadIdx.x] = img1[idx] * img2[idx];
        shared_norm1[threadIdx.x] = img1[idx] * img1[idx];
        shared_norm2[threadIdx.x] = img2[idx] * img2[idx];
    }
    else {
        shared_dot[threadIdx.x] = 0.0f;
        shared_norm1[threadIdx.x] = 0.0f;
        shared_norm2[threadIdx.x] = 0.0f;
    }

    __syncthreads();  // Synchronize threads within the block

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_dot[threadIdx.x] += shared_dot[threadIdx.x + s];
            shared_norm1[threadIdx.x] += shared_norm1[threadIdx.x + s];
            shared_norm2[threadIdx.x] += shared_norm2[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(dotProduct, shared_dot[0]);     // Atomic addition for dot product
        atomicAdd(norm1, shared_norm1[0]);       // Atomic addition for norm of img1
        atomicAdd(norm2, shared_norm2[0]);       // Atomic addition for norm of img2
    }
}

// Image Loading Function
void loadImage(const wchar_t* filename, unsigned char*& data, int& width, int& height, int& channels) {
    Bitmap bitmap(filename);
    if (bitmap.GetLastStatus() != Ok) {
        std::cerr << "Error: Failed to load image." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::wcout << L"Image Loaded: " << filename << std::endl;
    std::wcout << L"  Dimensions: " << bitmap.GetWidth() << L" x " << bitmap.GetHeight() << std::endl;

    PixelFormat format = bitmap.GetPixelFormat();
    std::wcout << L"  Pixel Format: ";
    switch (format) {
    case PixelFormat24bppRGB: std::wcout << L"24-bit RGB"; break;
    case PixelFormat32bppARGB: std::wcout << L"32-bit ARGB"; break;
    case PixelFormat8bppIndexed: std::wcout << L"8-bit Indexed"; break;
    default: std::wcout << L"Unknown Format"; break;
    }
    std::wcout << std::endl;

    width = bitmap.GetWidth();
    height = bitmap.GetHeight();
    channels = 3;

    data = new unsigned char[width * height * channels];

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Color color;
            bitmap.GetPixel(x, y, &color);
            int idx = (y * width + x) * channels;
            data[idx] = color.GetR();
            data[idx + 1] = color.GetG();
            data[idx + 2] = color.GetB();
        }
    }
}

// Function to compare images using MSE and Cosine Similarity
void compareImages(const unsigned char* h_img1, const unsigned char* h_img2, int width, int height, int channels) {
    const int imageSize = width * height * channels;
    unsigned char* d_img1, * d_img2;
    float* d_mse, * d_dot, * d_norm1, * d_norm2;

    cudaMalloc(&d_img1, imageSize);
    cudaMalloc(&d_img2, imageSize);
    cudaMalloc(&d_mse, sizeof(float));
    cudaMalloc(&d_dot, sizeof(float));
    cudaMalloc(&d_norm1, sizeof(float));
    cudaMalloc(&d_norm2, sizeof(float));

    cudaMemcpy(d_img1, h_img1, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, h_img2, imageSize, cudaMemcpyHostToDevice);

    cudaMemset(d_mse, 0, sizeof(float));
    cudaMemset(d_dot, 0, sizeof(float));
    cudaMemset(d_norm1, 0, sizeof(float));
    cudaMemset(d_norm2, 0, sizeof(float));

    const int blockSize = 256;
    const int gridSize = (imageSize + blockSize - 1) / blockSize;

    // Launch CUDA kernels
    calculateMSE << <gridSize, blockSize >> > (d_img1, d_img2, d_mse, imageSize);
    calculateCosine << <gridSize, blockSize >> > (d_img1, d_img2, d_dot, d_norm1, d_norm2, imageSize);

    // Copy results back to host
    float mse, dot, norm1, norm2;
    cudaMemcpy(&mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&dot, d_dot, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&norm1, d_norm1, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&norm2, d_norm2, sizeof(float), cudaMemcpyDeviceToHost);

    mse /= imageSize;
    float cosine_similarity = dot / (sqrt(norm1) * sqrt(norm2));

    std::cout << "MSE: " << mse << std::endl;
    std::cout << "Cosine Similarity: " << cosine_similarity << std::endl;

    // Cleanup
    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_mse);
    cudaFree(d_dot);
    cudaFree(d_norm1);
    cudaFree(d_norm2);
}

// Main function to load images and compare them
int main() {
    ULONG_PTR gdiplusToken;
    GdiplusStartupInput gdiplusStartupInput;
    GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

    int width1, height1, channels1;
    int width2, height2, channels2;
    int width3, height3, channels3;
    int width4, height4, channels4;
    int width5, height5, channels5;
    int width6, height6, channels6;
    int width7, height7, channels7;
    int width8, height8, channels8;
    int width9, height9, channels9;
    int width10, height10, channels10;
    int width11, height11, channels11;
    int width12, height12, channels12;
    int width13, height13, channels13;
    int width14, height14, channels14;
    int width15, height15, channels15;
    int width16, height16, channels16;
    int width17, height17, channels17;
    int width18, height18, channels18;
    int width19, height19, channels19;
    unsigned char* h_img1, * h_img2, * h_img3, * h_img4, * h_img5, * h_img6, * h_img7, * h_img8, * h_img9, * h_img10, 
        * h_img11, * h_img12, * h_img13, * h_img14, * h_img15, * h_img16, * h_img17, * h_img18, * h_img19;

    // Load images
    loadImage(L"1.png", h_img1, width1, height1, channels1);
    loadImage(L"2.png", h_img2, width2, height2, channels2);
    loadImage(L"3.png", h_img3, width3, height3, channels3);
    loadImage(L"4.png", h_img4, width4, height4, channels4);
    loadImage(L"5.png", h_img5, width5, height5, channels5);
    loadImage(L"6.png", h_img6, width6, height6, channels6);
    loadImage(L"7.png", h_img7, width7, height7, channels7);
    loadImage(L"8.png", h_img8, width8, height8, channels8);
    loadImage(L"9.png", h_img9, width9, height9, channels9);
    loadImage(L"10.png", h_img10, width10, height10, channels10);
    loadImage(L"11.png", h_img11, width11, height11, channels11);
    loadImage(L"12.png", h_img12, width12, height10, channels12);
    loadImage(L"13.png", h_img13, width13, height13, channels13);
    loadImage(L"14.png", h_img14, width14, height14, channels14);
    loadImage(L"15.png", h_img15, width15, height15, channels15);
    loadImage(L"16.png", h_img16, width16, height16, channels16);
    loadImage(L"17.png", h_img17, width17, height17, channels17);
    loadImage(L"18.png", h_img18, width18, height18, channels18);
    loadImage(L"19.png", h_img19, width19, height19, channels19);
    if (width1 != width2 || height1 != height2 || channels1 != channels2 
        || width1 != width3 || height1 != height3 || channels1 != channels3) {
        std::cerr << "Error: Image dimensions do not match." << std::endl;
        return EXIT_FAILURE;
    }

    // Compare images
    compareImages(h_img1, h_img2, width1, height1, channels1);
    compareImages(h_img1, h_img3, width1, height1, channels1);
    compareImages(h_img1, h_img4, width1, height1, channels1);
    compareImages(h_img1, h_img5, width1, height1, channels1);
    compareImages(h_img5, h_img6, width5, height5, channels5);
    compareImages(h_img5, h_img7, width5, height5, channels5);
    compareImages(h_img7, h_img13, width7, height7, channels7);
    compareImages(h_img7, h_img14, width7, height7, channels7);
    compareImages(h_img10, h_img11, width10, height10, channels10);
    compareImages(h_img10, h_img12, width10, height10, channels10);
    compareImages(h_img11, h_img12, width11, height11, channels11);
    compareImages(h_img13, h_img14, width13, height13, channels14);
    compareImages(h_img15, h_img16, width15, height15, channels15);
    compareImages(h_img15, h_img17, width15, height15, channels15);
    compareImages(h_img16, h_img17, width16, height16, channels16);
    compareImages(h_img18, h_img19, width18, height18, channels18);
    compareImages(h_img18, h_img8, width18, height18, channels18);
    compareImages(h_img9, h_img10, width9, height9, channels9);
    compareImages(h_img3, h_img9, width3, height3, channels3);
    compareImages(h_img11, h_img16, width11, height11, channels11);

    // Cleanup
    delete[] h_img1;
    delete[] h_img2;
    delete[] h_img3;
    delete[] h_img4;
    delete[] h_img5;
    delete[] h_img6;
    delete[] h_img7;
    delete[] h_img8;
    delete[] h_img9;
    delete[] h_img10;
    delete[] h_img11;

    GdiplusShutdown(gdiplusToken);

    return 0;
}
