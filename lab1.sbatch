#!/bin/bash
#SBATCH --job-name=lab1_sin
#SBATCH --output=lab1.%j.out
#SBATCH --error=lab1.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
# If your cluster accepts a partition for GPU compute, uncomment and set it:
# #SBATCH --partition=gpu
# If your Slurm supports generic resources 'gres=gpu:1' and it is required, uncomment:
# #SBATCH --gres=gpu:1

# --- Environment setup: try module load, fallback to /usr/local/cuda ---
echo "===== SLURM job start: $(date) on $(hostname) ====="
if command -v module >/dev/null 2>&1; then
  module purge
  module load cuda/12.6 2>/dev/null || module load cuda/12.5 2>/dev/null || module load cuda/11.8 2>/dev/null || true
fi

# If nvcc not found, try fallback path (adjust if CUDA installed elsewhere)
if ! command -v nvcc >/dev/null 2>&1; then
  echo "No nvcc in PATH, prepending /usr/local/cuda/bin (fallback)"
  export PATH=/usr/local/cuda/bin:${PATH}
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
fi

echo "nvcc version:"
nvcc --version

echo "nvidia-smi:"
nvidia-smi

# --- Compile: include support for V100 (sm_70) and newer sm_75 as fallback ---
NVCC_FLAGS="-O3 -lineinfo -Xcompiler -fPIC \
 -gencode=arch=compute_70,code=sm_70 \
 -gencode=arch=compute_75,code=sm_75"

echo "Compiling lab1.cu -> lab1"
nvcc $NVCC_FLAGS lab1.cu -o lab1 || { echo "Compilation failed"; exit 10; }

# --- Run: pass N (optional) ---
N_ARG=${1:-10000000}
echo "Running binary with N=${N_ARG}"
./lab1 ${N_ARG}

echo "===== SLURM job end: $(date) ====="