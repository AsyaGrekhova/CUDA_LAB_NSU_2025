#!/bin/bash
#SBATCH --job-name=img_filters
#SBATCH --output=lab_filters.%j.out
#SBATCH --error=lab_filters.%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# NOTE: do NOT force a gres line here if your cluster rejects it.
# If your cluster uses a GPU partition, set --partition accordingly:
# #SBATCH --partition=gpuserv
# If your site requires GRES, adjust to valid value, e.g. --gres=gpu:1

echo "===== SLURM job start: $(date) on $(hostname) ====="

# try to load usual cuda module; if unavailable, prepend /usr/local/cuda/bin
module purge
if module avail cuda 2>&1 | grep -q cuda; then
    module load cuda || true
else
    echo "CUDA module not found via 'module'; prepending /usr/local/cuda/bin to PATH (fallback)"
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi

echo "nvcc version:"
nvcc --version || true
echo "nvidia-smi:"
nvidia-smi || true

# compile
SRC=lab_filters.cu
EXE=lab_filters
echo "Compiling $SRC -> $EXE"
nvcc -O3 -arch=sm_70 -o ${EXE} ${SRC} -lpng || { echo "Compilation failed"; exit 2; }

# run (provide image path as first argument); default will use input.png (script creates test image)
INPUT=${1:-input.png}
echo "Running binary with input = ${INPUT}"
# use srun to schedule on allocated node (if cluster supports GPUs, srun will use them)
srun --mpi=none ./${EXE} "${INPUT}"

echo "===== SLURM job end: $(date) ====="
