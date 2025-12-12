// lab1.cu
// Исследование точности/производительности вычисления sin на GPU.
// Реализованы: float sinf, fast __sinf, double sin, lookup-table (constant memory).


#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdlib>

#define CHECK_CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(e) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(EXIT_FAILURE); } } while(0)

static const double PI = 3.14159265358979323846;

// constant lookup tables (360 values)
__constant__ float d_sin_table_f[360];
__constant__ double d_sin_table_d[360];

// --- kernels ---------------------------------------------------------------
__global__ void gpu_sin_float_sinf(float* arr, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float angle_deg = (float)(idx % 360);
    float angle_rad = angle_deg * (3.14159265358979323846f / 180.0f);
    arr[idx] = sinf(angle_rad);
}

__global__ void gpu_sin_float_fast(float* arr, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float angle_deg = (float)(idx % 360);
    float angle_rad = angle_deg * (3.14159265358979323846f / 180.0f);
    arr[idx] = __sinf(angle_rad);
}

__global__ void gpu_sin_double(double* arr, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double angle_deg = (double)(idx % 360);
    double angle_rad = angle_deg * (PI / 180.0);
    arr[idx] = sin(angle_rad);
}

// lookup-table kernels using constant memory
__global__ void gpu_table_float(float* out, size_t n) {
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * (size_t)blockDim.x;
    int r = (int)(tid % 360);
    for (size_t i = tid; i < n; i += stride) {
        out[i] = d_sin_table_f[r];
        if (++r == 360) r = 0;
    }
}

__global__ void gpu_table_double(double* out, size_t n) {
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * (size_t)blockDim.x;
    int r = (int)(tid % 360);
    for (size_t i = tid; i < n; i += stride) {
        out[i] = d_sin_table_d[r];
        if (++r == 360) r = 0;
    }
}

// --- host error computations ------------------------------------------------
double calculate_error_float(const float* data, size_t n) {
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double exact = std::sin((double)(i % 360) * PI / 180.0);
        total += std::abs(exact - (double)data[i]);
    }
    return total / (double)n;
}

double calculate_error_double(const double* data, size_t n) {
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double exact = std::sin((double)(i % 360) * PI / 180.0);
        total += std::abs(exact - data[i]);
    }
    return total / (double)n;
}

// --- main ------------------------------------------------------------------
int main(int argc, char** argv) {
    // N можно передать аргументом; по умолчанию 10M
    uint64_t N = (argc > 1) ? std::stoull(argv[1]) : 10000000ULL;
    const int BLOCK_SIZE = 256;

    // Проверка наличия устройств
    int deviceCount = 0;
    cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
    if (devErr != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA devices found or failed to query devices: "
                  << cudaGetErrorString(devErr) << std::endl;
        return 1;
    }
    // Используем устройство 0 по умолчанию
    CHECK_CUDA(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    std::cout << "ИССЛЕДОВАНИЕ ТОЧНОСТИ СИНУСА НА " << prop.name << std::endl;
    std::cout << "CUDA device properties: maxGridDimX=" << prop.maxGridSize[0]
              << " maxThreadsPerBlock=" << prop.maxThreadsPerBlock << std::endl;
    std::cout << "========================================\n\n";
    std::cout << "Запуск вычислений на GPU...\n\n";

    // Подготовка lookup-table на хосте и копирование в constant память
    float h_tab_f[360];
    double h_tab_d[360];
    for (int i = 0; i < 360; ++i) {
        double ang = (double)i * PI / 180.0;
        double v = std::sin(ang);
        h_tab_d[i] = v;
        h_tab_f[i] = (float)v;
    }
    CHECK_CUDA(cudaMemcpyToSymbol(d_sin_table_f, h_tab_f, sizeof(h_tab_f)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_sin_table_d, h_tab_d, sizeof(h_tab_d)));

    // Уточняем доступную память и при необходимости уменьшаем N, чтобы избежать OOM
    size_t freeMem = 0, totalMem = 0;
    CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    // резервируем 10% памяти для ОС/других целей
    size_t safeFree = (size_t)((double)freeMem * 0.9);
    uint64_t bytes_needed = N * sizeof(float); // для float массива
    if (bytes_needed > safeFree) {
        uint64_t max_fit = safeFree / sizeof(float);
        if (max_fit < 1024) {
            std::cerr << "Too little free device memory to allocate arrays. freeMem=" << freeMem << std::endl;
            return 1;
        }
        std::cerr << "Reducing N from " << N << " to " << max_fit << " due to device memory constraints\n";
        N = max_fit;
    }

    // Вычисляем grid, не превышая аппаратный максимум prop.maxGridSize[0]
    uint64_t rawGrid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint64_t grid64 = rawGrid;
    if (grid64 > (uint64_t)prop.maxGridSize[0]) grid64 = (uint64_t)prop.maxGridSize[0];
    int GRID = (int)grid64;
    if (GRID < 1) GRID = 1;

    // Создадим события для измерения времени
    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));

    // containers
    float t_sinf = 0.0f, t_fast = 0.0f, t_double = 0.0f, t_table = 0.0f;
    double err_sinf = 0.0, err_fast = 0.0, err_double = 0.0, err_table = 0.0;

    const int KPRINT = 5;
    std::vector<float> first_sinf(KPRINT), first_fast(KPRINT), first_table(KPRINT);
    std::vector<double> first_double(KPRINT);

    // --- 1) float sinf ---
    {
        size_t bytes = (size_t)N * sizeof(float);
        float* d_arr = nullptr;
        if (cudaMalloc(&d_arr, bytes) != cudaSuccess) { std::cerr<<"OOM allocating float array\n"; return 1; }

        CHECK_CUDA(cudaEventRecord(t0));
        gpu_sin_float_sinf<<<GRID, BLOCK_SIZE>>>(d_arr, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        CHECK_CUDA(cudaEventElapsedTime(&t_sinf, t0, t1));

        std::vector<float> h_res(N);
        CHECK_CUDA(cudaMemcpy(h_res.data(), d_arr, bytes, cudaMemcpyDeviceToHost));
        err_sinf = calculate_error_float(h_res.data(), N);
        for (int i = 0; i < KPRINT && i < (int)N; ++i) first_sinf[i] = h_res[i];
        CHECK_CUDA(cudaFree(d_arr));
    }

    // --- 2) float __sinf ---
    {
        size_t bytes = (size_t)N * sizeof(float);
        float* d_arr = nullptr;
        CHECK_CUDA(cudaMalloc(&d_arr, bytes));

        CHECK_CUDA(cudaEventRecord(t0));
        gpu_sin_float_fast<<<GRID, BLOCK_SIZE>>>(d_arr, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        CHECK_CUDA(cudaEventElapsedTime(&t_fast, t0, t1));

        std::vector<float> h_res(N);
        CHECK_CUDA(cudaMemcpy(h_res.data(), d_arr, bytes, cudaMemcpyDeviceToHost));
        err_fast = calculate_error_float(h_res.data(), N);
        for (int i = 0; i < KPRINT && i < (int)N; ++i) first_fast[i] = h_res[i];
        CHECK_CUDA(cudaFree(d_arr));
    }

    // --- 3) double sin ---
    {
        size_t bytes = (size_t)N * sizeof(double);
        double* d_arr = nullptr;
        // если памяти не хватает под double, уменьшаем N вдвое и пробуем снова
        if (bytes > freeMem) {
            uint64_t max_double = freeMem / sizeof(double);
            if (max_double < 1024) {
                std::cerr << "Not enough memory for double run; skipping double test\n";
            } else {
                std::cerr << "Reducing N for double from " << N << " to " << max_double << "\n";
                N = max_double;
                bytes = (size_t)N * sizeof(double);
            }
        }
        if (cudaMalloc(&d_arr, bytes) == cudaSuccess) {
            CHECK_CUDA(cudaEventRecord(t0));
            gpu_sin_double<<<GRID, BLOCK_SIZE>>>(d_arr, N);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaEventRecord(t1));
            CHECK_CUDA(cudaEventSynchronize(t1));
            CHECK_CUDA(cudaEventElapsedTime(&t_double, t0, t1));

            std::vector<double> h_res(N);
            CHECK_CUDA(cudaMemcpy(h_res.data(), d_arr, bytes, cudaMemcpyDeviceToHost));
            err_double = calculate_error_double(h_res.data(), N);
            for (int i = 0; i < KPRINT && i < (int)N; ++i) first_double[i] = h_res[i];
            CHECK_CUDA(cudaFree(d_arr));
        } else {
            std::cerr << "Skipping double kernel due to allocation failure\n";
        }
    }

    // --- 4) lookup-table (float) ---
    {
        size_t bytes = (size_t)N * sizeof(float);
        float* d_arr = nullptr;
        CHECK_CUDA(cudaMalloc(&d_arr, bytes));

        CHECK_CUDA(cudaEventRecord(t0));
        gpu_table_float<<<GRID, BLOCK_SIZE>>>(d_arr, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        CHECK_CUDA(cudaEventElapsedTime(&t_table, t0, t1));

        std::vector<float> h_res(N);
        CHECK_CUDA(cudaMemcpy(h_res.data(), d_arr, bytes, cudaMemcpyDeviceToHost));
        err_table = calculate_error_float(h_res.data(), N);
        for (int i = 0; i < KPRINT && i < (int)N; ++i) first_table[i] = h_res[i];
        CHECK_CUDA(cudaFree(d_arr));
    }

    // --- print results in requested style ---
    std::cout << "РЕЗУЛЬТАТЫ ДЛЯ " << N << " ЭЛЕМЕНТОВ:\n";
    std::cout << "========================================\n";
    std::cout << "ФУНКЦИЯ      | ВРЕМЯ (ms) | ОШИБКА\n";
    std::cout << "-------------|------------|-----------------\n";
    std::cout << std::left << std::setw(13) << "float sinf" << "| "
              << std::right << std::setw(10) << std::fixed << std::setprecision(6) << t_sinf << " | "
              << std::scientific << std::setprecision(6) << err_sinf << "\n";
    std::cout << std::left << std::setw(13) << "float __sinf" << "| "
              << std::right << std::setw(10) << std::fixed << std::setprecision(6) << t_fast << " | "
              << std::scientific << std::setprecision(6) << err_fast << "\n";
    std::cout << std::left << std::setw(13) << "double sin" << "| "
              << std::right << std::setw(10) << std::fixed << std::setprecision(6) << t_double << " | "
              << std::scientific << std::setprecision(6) << err_double << "\n";
    std::cout << std::left << std::setw(13) << "float table" << "| "
              << std::right << std::setw(10) << std::fixed << std::setprecision(6) << t_table << " | "
              << std::scientific << std::setprecision(6) << err_table << "\n\n";

    std::cout << "ПРОВЕРКА КОРРЕКТНОСТИ:\n";
    std::cout << "Первые 5 значений float sinf: ";
    for (int i = 0; i < KPRINT; ++i) {
        std::cout << std::scientific << std::setprecision(6) << first_sinf[i] << " ";
    }
    std::cout << "\nПервые 5 значений float table: ";
    for (int i = 0; i < KPRINT; ++i) {
        std::cout << std::scientific << std::setprecision(6) << first_table[i] << " ";
    }
    std::cout << "\n\nGPU вычисления работают: " << ((KPRINT > 0 && std::abs(first_sinf[1]) > 0.0f) ? "ДА" : "НЕТ") << "\n\n";

    std::cout << "СРАВНЕНИЕ С CPU (первые 3 значения):\n";
    std::cout << "i\tGPU sinf\tCPU\t\tРазница\n";
    for (int i = 0; i < 3; ++i) {
        float gpu_val = first_sinf[i];
        float cpu_val = sinf((float)(i % 360) * (float)(PI / 180.0));
        float diff = fabsf(gpu_val - cpu_val);
        std::cout << i << "\t" << std::scientific << std::setprecision(6) << gpu_val
                  << "\t" << cpu_val << "\t" << diff << "\n";
    }

    std::cout << "\nВЫВОДЫ:\n";
    std::cout << "- sinf: баланс скорости и точности\n";
    std::cout << "- __sinf: быстрее, но менее точен\n";
    std::cout << "- double: максимальная точность, но медленнее\n";
    std::cout << "- lookup-table: минимизирует вычисления sin и даёт стабильную точность\n\n";

    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));
    return 0;
}