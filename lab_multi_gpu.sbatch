#!/bin/bash
#SBATCH -J lab_multi_gpu
#SBATCH -p gpuserv
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:10:00
#SBATCH -o lab_multi_gpu.%j.out
#SBATCH -e lab_multi_gpu.%j.err

module load nvidia/cuda

echo "===== SLURM job start ====="
date
echo "Running on node: $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "Cores: $SLURM_CPUS_PER_TASK"

echo "nvcc version:"
nvcc --version

echo "nvidia-smi:"
nvidia-smi

echo "Compiling..."
nvcc lab_multi_gpu.cu -lpng -o lab_multi_gpu

echo "Running..."
./lab_multi_gpu input2.png output.png