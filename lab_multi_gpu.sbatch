#!/bin/bash
#SBATCH --job-name=lab_multi_gpu
#SBATCH --output=lab_multi_gpu.out
#SBATCH --error=lab_multi_gpu.err
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# NOTE: many clusters require a specific gres string; earlier you had "Invalid generic resource".
# If your cluster supports it and you know the correct value, uncomment and adjust:
# SBATCH --gres=gpu:2

set -euo pipefail

echo "===== SBATCH: starting job on $(hostname) at $(date) ====="
echo ""

# 1) Try to load CUDA module (multiple common names). If module is not available, continue.
echo "Attempting to load CUDA module..."
module load cuda/12.6 2>/dev/null || module load cuda/12.5 2>/dev/null || module load cuda/11.8 2>/dev/null || true

# 2) Ensure nvcc is in PATH (fallback to /usr/local/cuda/bin if needed)
if ! command -v nvcc >/dev/null 2>&1; then
  echo "No nvcc in PATH, prepending /usr/local/cuda/bin (fallback)"
  export PATH=/usr/local/cuda/bin:$PATH
fi

echo "nvcc version:"
nvcc --version || true
echo ""

echo "nvidia-smi:"
nvidia-smi || true
echo ""

# 3) Ensure source files exist in working directory
WORKDIR="$(pwd)"
echo "Working dir: $WORKDIR"
echo ""

# 4) Download stb headers (try multiple CDNs). If both fail, exit with informative message.
download_stb() {
  local fname="$1"
  shift
  local urls=("$@")
  for url in "${urls[@]}"; do
    echo "Trying to download $fname from: $url"
    # prefer curl, fallback to wget
    if command -v curl >/dev/null 2>&1; then
      if curl -fsSL -o "$fname" "$url"; then
        echo "Downloaded $fname from $url"
        return 0
      else
        echo "curl failed for $url"
      fi
    fi
    if command -v wget >/dev/null 2>&1; then
      if wget -q -O "$fname" "$url"; then
        echo "Downloaded $fname from $url (wget)"
        return 0
      else
        echo "wget failed for $url"
      fi
    fi
  done
  return 1
}

# candidate URLs (jsDelivr CDN, GitHub raw). Add/remove if needed.
STB_URLS=(
  "https://cdn.jsdelivr.net/gh/nothings/stb/stb_image.h"
  "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h"
)
STB_WRITE_URLS=(
  "https://cdn.jsdelivr.net/gh/nothings/stb/stb_image_write.h"
  "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h"
)

# Download only if missing
if [ ! -f stb_image.h ]; then
  if ! download_stb stb_image.h "${STB_URLS[@]}"; then
    echo "ERROR: failed to download stb_image.h from all candidates."
    echo "Please place stb_image.h into the job directory and re-run."
    exit 2
  fi
else
  echo "stb_image.h already present, skipping download."
fi

if [ ! -f stb_image_write.h ]; then
  if ! download_stb stb_image_write.h "${STB_WRITE_URLS[@]}"; then
    echo "WARNING: failed to download stb_image_write.h from all candidates."
    echo "You can either (a) place stb_image_write.h into the job directory manually, or"
    echo "(b) edit this script to use libpng-based I/O instead."
    # keep going (optionally you can exit here). We'll exit so compilation doesn't fail unexpectedly:
    exit 3
  fi
else
  echo "stb_image_write.h already present, skipping download."
fi

echo ""
ls -l stb_image.h stb_image_write.h || true
echo ""

# 5) Compile the CUDA program
SRC="lab_multi_gpu.cu"
BIN="lab_multi_gpu"

if [ ! -f "$SRC" ]; then
  echo "ERROR: source file $SRC not found in $WORKDIR"
  exit 4
fi

echo "Compiling $SRC -> $BIN"
# compile with reasonable arch support; add -arch flags if you need specific SMs
nvcc -O3 -lineinfo -std=c++14 -arch=sm_70 -o "$BIN" "$SRC" -lpng || {
  echo "nvcc compile failed; trying fallback without -lpng"
  nvcc -O3 -lineinfo -std=c++14 -arch=sm_70 -o "$BIN" "$SRC" || { echo "Compilation failed"; exit 5; }
}

echo "Compilation finished successfully."
echo ""

# 6) Run the binary (pass input2.png). Use srun so Slurm tracks the process.
INPUT="input2.png"
if [ ! -f "$INPUT" ]; then
  echo "Input image $INPUT not found in working directory ($WORKDIR)."
  echo "Place $INPUT into the job directory and re-run the job."
  exit 6
fi

echo "Running the program: ./$BIN $INPUT"
echo "----- PROGRAM OUTPUT START -----"
# run with srun so resources / cgroups are applied; --mpi=none to avoid mpiexec wrapper on some clusters
srun --mpi=none ./"$BIN" "$INPUT"
RET=$?
echo "----- PROGRAM OUTPUT  END  -----"
echo "Program exit code: $RET"

echo ""
echo "===== SBATCH: job finished at $(date) ====="
exit $RET