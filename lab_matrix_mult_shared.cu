#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define BLOCK_SIZE 16
#define BASE_TYPE double

__global__ void matrixMultShared(
    const BASE_TYPE *A,
    const BASE_TYPE *B,
    BASE_TYPE *C,
    int Acols,
    int Bcols)
{
    int aBegin = Acols * blockDim.y * blockIdx.y;
    int aEnd   = aBegin + Acols - 1;
    int aStep  = blockDim.x;

    int bBegin = blockDim.x * blockIdx.x;
    int bStep  = blockDim.y * Bcols;

    __shared__ BASE_TYPE As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ BASE_TYPE Bs[BLOCK_SIZE][BLOCK_SIZE];

    BASE_TYPE sum = 0.0;

    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
    {
        As[threadIdx.y][threadIdx.x] =
            A[ia + Acols * threadIdx.y + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] =
            B[ib + Bcols * threadIdx.y + threadIdx.x];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    int idx =
        Bcols * (blockDim.y * blockIdx.y + threadIdx.y) +
        blockDim.x * blockIdx.x + threadIdx.x;

    C[idx] = sum;
}

int toMultiple(int a, int b)
{
    int mod = a % b;
    if (mod != 0) a += (b - mod);
    return a;
}

int main()
{
    int Arows = 1000;
    int Acols = 2000;
    int Brows = Acols;
    int Bcols = 1500;

    Arows = toMultiple(Arows, BLOCK_SIZE);
    Acols = toMultiple(Acols, BLOCK_SIZE);
    Brows = toMultiple(Brows, BLOCK_SIZE);
    Bcols = toMultiple(Bcols, BLOCK_SIZE);

    printf("Arows = %d\n", Arows);
    printf("Acols = %d\n", Acols);
    printf("Brows = %d\n", Brows);
    printf("Bcols = %d\n", Bcols);

    size_t Asize = Arows * Acols * sizeof(BASE_TYPE);
    size_t Bsize = Brows * Bcols * sizeof(BASE_TYPE);
    size_t Csize = Arows * Bcols * sizeof(BASE_TYPE);

    BASE_TYPE *h_A = (BASE_TYPE*)malloc(Asize);
    BASE_TYPE *h_B = (BASE_TYPE*)malloc(Bsize);
    BASE_TYPE *h_C = (BASE_TYPE*)malloc(Csize);

    for (int i = 0; i < Arows * Acols; i++)
        h_A[i] = rand() / (BASE_TYPE)RAND_MAX;

    for (int i = 0; i < Brows * Bcols; i++)
        h_B[i] = rand() / (BASE_TYPE)RAND_MAX;

    BASE_TYPE *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, Asize);
    cudaMalloc((void**)&d_B, Bsize);
    cudaMalloc((void**)&d_C, Csize);

    cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(Bcols / BLOCK_SIZE, Arows / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMultShared<<<blocks, threads>>>(d_A, d_B, d_C, Acols, Bcols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernelTime;
    cudaEventElapsedTime(&kernelTime, start, stop);

    std::cout << "Kernel Time (shared memory): "
              << kernelTime << " ms\n";

    cudaMemcpy(h_C, d_C, Csize, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}