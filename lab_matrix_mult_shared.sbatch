#!/bin/bash
#SBATCH -J lab_matrix_mult_shared
#SBATCH -p gpuserv
#SBATCH -o lab_matrix_mult_shared.%j.out
#SBATCH -e lab_matrix_mult_shared.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:10:00

module load nvidia/cuda

echo "===== SLURM job start ====="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "JobID: $SLURM_JOBID"
echo "Cores: $SLURM_CPUS_PER_TASK"

echo "nvcc version:"
nvcc --version

echo "nvidia-smi:"
nvidia-smi

echo "Compiling..."
nvcc lab_matrix_mult_shared.cu -o lab_matrix_mult_shared.out

echo "Running..."
./lab_matrix_mult_shared.out

echo "===== SLURM job end ====="