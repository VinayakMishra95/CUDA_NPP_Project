#!/usr/bin/env bash
# ==============================================================================
# Execution Script for CUDA & NPP Enterprise Image Processing Pipeline
# Course: CUDA at Scale for the Enterprise
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "======================================================================"
echo " Starting CUDA at Scale Enterprise Project Execution Pipeline"
echo "======================================================================"
echo "Working directory: ${SCRIPT_DIR}"
date

# 1. Directory Setup
mkdir -p bin build data/input/batch_small output/large output/batch_small artifacts

# 2. Check Dataset Availability
if [ ! -f "data/input/sipi_5.2.01_zoneplate512.pgm" ]; then
    echo ">> Generating USC-SIPI benchmark dataset..."
    python3 scripts/prepare_dataset.py --output_dir data/input --num_small 100
fi

# 3. Detect Toolchain and Compile
if command -v nvcc >/dev/null 2>&1; then
    echo ">> NVIDIA CUDA Compiler (nvcc) detected. Building GPU release target..."
    make all -j4
    RUN_BINARY="./bin/cuda_image_proc"
else
    echo ">> NVCC not detected on current host. Building host verification target..."
    make sim HOST_CXX=clang++ -j4 || make sim HOST_CXX=g++ -j4
    RUN_BINARY="./bin/sim_image_proc"
fi

echo "======================================================================"
echo " Stage 1: Processing Tens of Large USC-SIPI Images (512x512 - 1024x1024)"
echo "======================================================================"
${RUN_BINARY} --input_dir data/input --output_dir output/large --filter all --benchmark 2>&1 | tee artifacts/execution_log.txt

echo "======================================================================"
echo " Stage 2: Processing Hundreds of Small Images (100x 128x128 Stream Batch)"
echo "======================================================================"
${RUN_BINARY} --input_dir data/input/batch_small --output_dir output/batch_small --filter all --benchmark 2>&1 | tee -a artifacts/execution_log.txt

echo "======================================================================"
echo " Stage 3: Packaging Evidence and Proof of Execution Artifacts"
echo "======================================================================"
tar -czvf artifacts/execution_artifacts.tar.gz \
    artifacts/execution_log.txt \
    artifacts/benchmark_report.md \
    output/large/*.pgm \
    2>&1 | tail -n 15

zip -r artifacts/execution_artifacts.zip \
    artifacts/execution_log.txt \
    artifacts/benchmark_report.md \
    output/large/*.pgm \
    >/dev/null 2>&1

echo "======================================================================"
echo " Execution and Packaging Completed Successfully!"
echo " Artifacts generated at:"
echo "   - artifacts/execution_log.txt"
echo "   - artifacts/benchmark_report.md"
echo "   - artifacts/execution_artifacts.tar.gz"
echo "   - artifacts/execution_artifacts.zip"
echo "======================================================================"
