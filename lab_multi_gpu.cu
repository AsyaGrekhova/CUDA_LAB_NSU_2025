#include <png.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>

#define BLOCK 16

// Простое размытие 3x3 (пример фильтра)
__global__
void blurKernel(const unsigned char* input, unsigned char* output,
                int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int out_idx = (y * width + x) * channels;

    for (int c = 0; c < channels; c++) {
        int sum = 0;
        int count = 0;

        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx;
                int ny = y + dy;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int in_idx = (ny * width + nx) * channels + c;
                    sum += input[in_idx];
                    count++;
                }
            }

        output[out_idx + c] = sum / count;
    }
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: ./lab_multi_gpu input2.png output.png\n";
        return 1;
    }

    const char* inFile  = argv[1];
    const char* outFile = argv[2];

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    std::cout << "deviceCount: " << deviceCount << std::endl;

    if (deviceCount < 2) {
        std::cerr << "Error: Need at least 2 GPUs\n";
        return 1;
    }

    // === ЧТЕНИЕ PNG ===
    png_image image;
    memset(&image, 0, sizeof(image));
    image.version = PNG_IMAGE_VERSION;

    if (!png_image_begin_read_from_file(&image, inFile)) {
        std::cerr << "PNG read error: " << image.message << std::endl;
        return 1;
    }

    image.format = PNG_FORMAT_RGBA;
    size_t imgSize = PNG_IMAGE_SIZE(image);
    unsigned char* host_img = (unsigned char*)malloc(imgSize);

    if (!png_image_finish_read(&image, NULL, host_img, 0, NULL)) {
        std::cerr << "PNG finish read error\n";
        free(host_img);
        return 1;
    }

    int width  = image.width;
    int height = image.height;
    int channels = 4;

    // === РАЗБИЕНИЕ ИЗОБРАЖЕНИЯ ===
    int half = height / 2;
    int gpu0_H = half + 1;              // + halo
    int gpu1_H = height - (half - 1);   // + halo

    size_t bytes0 = width * gpu0_H * channels;
    size_t bytes1 = width * gpu1_H * channels;

    unsigned char* ptr0 = host_img;
    unsigned char* ptr1 = host_img + (half - 1) * width * channels;

    dim3 block(BLOCK, BLOCK);

    // === GPU 0 ===
    cudaSetDevice(0);
    unsigned char *d_in0, *d_out0;
    cudaMalloc(&d_in0, bytes0);
    cudaMalloc(&d_out0, bytes0);
    cudaMemcpy(d_in0, ptr0, bytes0, cudaMemcpyHostToDevice);

    dim3 grid0((width + BLOCK - 1) / BLOCK,
               (gpu0_H + BLOCK - 1) / BLOCK);

    blurKernel<<<grid0, block>>>(d_in0, d_out0, width, gpu0_H, channels);
    cudaMemcpy(ptr0,
               d_out0,
               width * half * channels,
               cudaMemcpyDeviceToHost);

    // === GPU 1 ===
    cudaSetDevice(1);
    unsigned char *d_in1, *d_out1;
    cudaMalloc(&d_in1, bytes1);
    cudaMalloc(&d_out1, bytes1);
    cudaMemcpy(d_in1, ptr1, bytes1, cudaMemcpyHostToDevice);

    dim3 grid1((width + BLOCK - 1) / BLOCK,
               (gpu1_H + BLOCK - 1) / BLOCK);

    blurKernel<<<grid1, block>>>(d_in1, d_out1, width, gpu1_H, channels);
    cudaMemcpy(host_img + half * width * channels,
               d_out1 + width * channels,
               width * (height - half) * channels,
               cudaMemcpyDeviceToHost);

    // === ЗАПИСЬ PNG ===
    if (!png_image_write_to_file(&image, outFile, 0, host_img, 0, NULL)) {
        std::cerr << "PNG write error: " << image.message << std::endl;
    }

    std::cout << "Saved: " << outFile << std::endl;

    // === ОСВОБОЖДЕНИЕ ПАМЯТИ ===
    cudaSetDevice(0);
    cudaFree(d_in0);
    cudaFree(d_out0);

    cudaSetDevice(1);
    cudaFree(d_in1);
    cudaFree(d_out1);

    free(host_img);

    return 0;
}