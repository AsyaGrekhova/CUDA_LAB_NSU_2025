#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>

// ----------------------------
// Простая структура изображения
// ----------------------------
struct Image {
    int w, h;
    std::vector<unsigned char> data;

    Image(int width = 0, int height = 0)
        : w(width), h(height), data(width * height * 4, 0) {}
};

// ----------------------------
// Загрузка PNG через stb_image
// ----------------------------
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ----------------------------
// CUDA-ядро простого блюра 3x3
// ----------------------------
__global__
void blur_kernel(unsigned char* in, unsigned char* out, int w, int h, int offset_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int gy = y + offset_y;  // глобальная координата по Y

    if (x >= w || gy >= h) return;

    int4 sum = {0,0,0,0};
    int count = 0;

    for (int dy=-1; dy<=1; dy++)
    for (int dx=-1; dx<=1; dx++) {
        int nx = x + dx;
        int ny = gy + dy;
        if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
            int idx = (ny * w + nx) * 4;
            sum.x += in[idx + 0];
            sum.y += in[idx + 1];
            sum.z += in[idx + 2];
            sum.w += in[idx + 3];
            count++;
        }
    }

    int out_idx = (gy * w + x) * 4;
    out[out_idx + 0] = sum.x / count;
    out[out_idx + 1] = sum.y / count;
    out[out_idx + 2] = sum.z / count;
    out[out_idx + 3] = sum.w / count;
}

// -------------------------------------
// Фильтрация на одном GPU
// -------------------------------------
void run_on_device(int dev,
                   unsigned char* full_in,
                   unsigned char* full_out,
                   int w, int h,
                   int y0, int y1)
{
    cudaSetDevice(dev);

    int H = y1 - y0;
    int bytes = w * H * 4;

    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, full_in + y0 * w * 4, bytes, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid( (w+15)/16, (H+15)/16 );

    blur_kernel<<<grid,block>>>(d_in, d_out, w, h, y0);
    cudaDeviceSynchronize();

    cudaMemcpy(full_out + y0 * w * 4, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

// -------------------------------------
// МОНОЛИТНАЯ ФУНКЦИЯ multi-GPU
// -------------------------------------
void process_multi_gpu(const char* in_file, const char* out_file)
{
    int w, h, ch;
    unsigned char* img = stbi_load(in_file, &w, &h, &ch, 4);
    if (!img) {
        printf("ERROR: cannot load input image %s\n", in_file);
        return;
    }

    size_t total_bytes = w * h * 4;
    std::vector<unsigned char> output(total_bytes, 0);

    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);

    if (gpu_count == 0) {
        printf("ERROR: no CUDA devices!\n");
        return;
    }

    printf("Detected %d GPUs\n", gpu_count);

    // Разбиваем изображение по горизонтали
    int slice = h / gpu_count;

    std::vector<cudaStream_t> streams(gpu_count);

    for (int d = 0; d < gpu_count; d++) {
        int y0 = d * slice;
        int y1 = (d == gpu_count - 1) ? h : (d + 1) * slice;

        run_on_device(d, img, output.data(), w, h, y0, y1);
    }

    stbi_write_png(out_file, w, h, 4, output.data(), w * 4);
    printf("Saved result to %s\n", out_file);

    stbi_image_free(img);
}

// -------------------------------------
int main(int argc, char** argv)
{
    const char* input = "input2.png";
    const char* output = "output_multi_gpu.png";

    if (argc >= 2) input = argv[1];
    if (argc >= 3) output = argv[2];

    printf("Using input: %s\n", input);
    process_multi_gpu(input, output);
    return 0;
}