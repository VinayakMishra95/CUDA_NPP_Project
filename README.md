# CUDA & NPP Enterprise Image Processing Pipeline
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.x%20%7C%2012.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![NPP](https://img.shields.io/badge/NVIDIA-NPP-76B900.svg)](https://docs.nvidia.com/cuda/npp/index.html)
[![Standard](https://img.shields.io/badge/C%2B%2B-14%2F17-orange.svg)](https://isocpp.org)
[![Style](https://img.shields.io/badge/Code%20Style-Google-blueviolet.svg)](https://google.github.io/styleguide/cppguide.html)

**Course**: *CUDA at Scale for the Enterprise*  
**Repository**: [https://github.com/PascaleCourseraCourses/CUDAatScaleForTheEnterpriseCourseProjectTemplate](https://github.com/PascaleCourseraCourses/CUDAatScaleForTheEnterpriseCourseProjectTemplate) (Project Implementation)  
**Dataset**: [USC Viterbi School of Engineering SIPI Image Database](https://sipi.usc.edu/database/database.php) (Volume 3 Miscellaneous & Volume 1 Textures)

---

## 1. Overview & Objectives

In modern enterprise computer vision, medical imaging, and satellite surveillance workflows, latency and throughput constraints require massive parallel processing at scale. This project implements a high-throughput, GPU-accelerated image processing engine designed to process **tens of large high-resolution images** (512x512 to 1024x1024) and **hundreds of stream-ingested small images** (128x128).

The project combines two parallel computing paradigms:
1. **NVIDIA Performance Primitives (NPP)**: Utilizing highly tuned vendor primitives for 2D spatial convolution (`nppiFilterBox`), directional gradient filtering (`nppiFilterSobelHoriz` & `nppiFilterSobelVert`), and segmentation thresholding (`nppiThreshold_LTValGTVal`).
2. **Custom Optimized CUDA Kernels**: Implementing 2D discrete Laplacian edge extraction and Unsharp Mask sharpening with **on-chip shared memory tiling (16x16 with 1-pixel halo borders)** to minimize global memory bandwidth bottlenecks.

---

## 2. Project Architecture & Directory Structure

The repository adheres strictly to enterprise software engineering standards and the Google C++ Style Guide:

```text
cuda_npp_project/
├── .gitignore                    # Git tracking ignore rules
├── LICENSE                       # Apache 2.0 Open Source License
├── INSTALL                       # Step-by-step multi-platform installation guide
├── Makefile                      # Cross-platform build automation (NVCC and Host simulation)
├── README.md                     # Project overview, architecture, and user guide
├── PROJECT_DESCRIPTION.md        # Comprehensive report for peer review
├── run.sh                        # Automated execution and packaging script
├── include/                      # C++ Header files (Google Style compliant)
│   ├── common.h                  # Error macros (CUDA_CHECK, NPP_CHECK), timers, metrics
│   ├── image_io.h                # Netpbm PGM/PPM format parser and serializer
│   ├── npp_processor.h           # NVIDIA NPP image processing wrapper class
│   └── custom_kernel.h           # Custom CUDA shared-memory kernel processor class
├── src/                          # Implementation source files
│   ├── image_io.cpp              # Robust Netpbm binary parser implementation
│   ├── npp_processor.cu          # Pitched buffer memory management & NPP filters
│   ├── custom_kernel.cu          # Shared-memory tiled 2D convolution CUDA kernels
│   └── main.cpp                  # CLI argument parser, batch orchestrator & profiler
├── data/
│   └── input/                    # Staged USC-SIPI benchmark dataset
│       ├── sipi_5.1.09_moon256.pgm
│       ├── sipi_5.2.01_zoneplate512.pgm
│       ├── sipi_5.3.01_aerial1024.pgm
│       └── batch_small/          # 100+ small 128x128 images for stream scaling
├── output/                       # Output directory for processed artifact images
│   ├── large/                    # Processed large benchmark images
│   └── batch_small/              # Processed batch images
├── scripts/
│   └── prepare_dataset.py        # Dataset downloader & procedural generator script
└── artifacts/                    # Execution verification proofs for peer grading
    ├── execution_log.txt         # Full terminal execution log
    ├── benchmark_report.md       # Detailed performance and latency breakdown
    ├── execution_artifacts.tar.gz# Compressed archive for Coursera peer submission
    └── execution_artifacts.zip   # Zip format archive
```

---

## 3. Dataset Specifications (USC-SIPI Database)

The primary dataset used is from the **USC Viterbi School of Engineering's SIPI Image Database** ([https://sipi.usc.edu/database/](https://sipi.usc.edu/database/)):

1. **Volume 3: Miscellaneous**:
   - `5.1.09` (Moon surface, 256x256 monochrome)
   - `5.1.10` (Aerial photo, 256x256 monochrome)
   - `5.1.11` (Airplane / F-16, 256x256 monochrome)
   - `5.1.12` (Clock face, 256x256 monochrome)
   - `5.1.13` (Resolution chart, 256x256 monochrome)
   - `5.1.14` (Chemical plant, 256x256 monochrome)
   - `5.2.01` (Fresnel Zone Plate test pattern, 512x512 monochrome)
   - `boat512`, `stream512`, `truck512` (Classic 512x512 scenes)
   - `5.3.01`, `5.3.02` (High-altitude aerial reconnaissance, 1024x1024)
   - `testpat1024` (High-frequency resolution pattern, 1024x1024)
2. **Batch Stream Dataset (`batch_small/`)**:
   - 100 distinct 128x128 images covering varying frequency spectra, synthetic terrain, and edge densities to evaluate pipeline latency under rapid request arrival rates.

The repository includes a self-contained ingest script `scripts/prepare_dataset.py` that stages the dataset automatically.

---

## 4. Build Instructions

### Prerequisites
- Linux (Ubuntu 20.04/22.04 LTS or Coursera GPU Lab environment) or macOS.
- NVIDIA CUDA Toolkit 11.0+ or 12.0+ (with `nvcc`, `libcudart`, `libnppc`, `libnppif`, `libnppig`, `libnppitc`).
- C++14/17 compliant compiler (`g++` 7.5+ or `clang++`).

### Building with NVCC (GPU Acceleration)
To compile the GPU binary using `nvcc` and link against NVIDIA NPP:
```bash
make all -j4
```
The compiled executable will be placed in `bin/cuda_image_proc`.

### Building Simulation Verification Target (Host Validation)
To compile and test on a development machine without an NVIDIA GPU:
```bash
make sim HOST_CXX=g++ -j4
# Or on macOS with Clang:
make sim HOST_CXX=clang++ -j4
```
The simulation executable will be placed in `bin/sim_image_proc`.

### Cleaning Build Artifacts
```bash
make clean
```

---

## 5. Execution & Command-Line Arguments

The application accepts a flexible set of command-line flags conforming to the assignment rubric:

```text
Usage: bin/cuda_image_proc [options]

Options:
  --input_dir <path>       Directory with input PGM/PPM images (default: data/input)
  --output_dir <path>      Directory for processed output images (default: output)
  --input_file <path>      Process a specific single image instead of directory
  --filter <name>          Filter operation to apply:
                           'box'              - NPP 2D Box Blur
                           'sobel'            - NPP 2D Sobel Edge Detection
                           'threshold'        - NPP Binarization Thresholding
                           'custom_laplacian' - Custom CUDA Shared-Mem Laplacian
                           'custom_sharpen'   - Custom CUDA Unsharp Sharpening
                           'all'              - Execute full pipeline (default)
  --threshold_val <0-255>  Cutoff for threshold filter (default: 128)
  --kernel_size <size>     Odd integer mask size for box filter (default: 3)
  --benchmark              Enable detailed timing and throughput logging
  --verbose                Enable verbose diagnostic output
  --device <id>            CUDA device ID (default: 0)
  --help                   Display this help message
```

### Execution Examples

1. **Full Automated Run (Large and Small Batches)**:
   ```bash
   ./run.sh
   ```

2. **Run All Filters on Large Dataset with Benchmark Metrics**:
   ```bash
   ./bin/cuda_image_proc --input_dir data/input --output_dir output/large --filter all --benchmark
   ```

3. **Process a Single Image with NPP Sobel Edge Detection**:
   ```bash
   ./bin/cuda_image_proc --input_file data/input/sipi_5.2.01_zoneplate512.pgm --output_dir output --filter sobel
   ```

4. **Run High-Throughput Batch Processing over 100 Small Images**:
   ```bash
   ./bin/cuda_image_proc --input_dir data/input/batch_small --output_dir output/batch_small --filter threshold --threshold_val 140
   ```

---

## 6. Proof of Execution Artifacts

Per the grading rubric, execution evidence is packaged in the `artifacts/` folder:
- **`artifacts/execution_log.txt`**: Complete console output detailing memory allocations, device query, per-image filter execution, latency breakdowns, and aggregated throughput metrics.
- **`artifacts/benchmark_report.md`**: Detailed quantitative performance analysis.
- **`artifacts/execution_artifacts.tar.gz`**: Gzip-compressed archive holding execution logs, benchmark reports, and processed output images.
- **`artifacts/execution_artifacts.zip`**: Standard zip archive holding execution artifacts.

---

## 7. Adherence to Google C++ Style Guide

The codebase follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html):
- **Naming Conventions**: `PascalCase` for types (`ImageBuffer`, `NppImageProcessor`), `kConstantName` for constants (`kTileDim`, `kHaloDim`), `snake_case` for local variables and functions, trailing underscore for private member variables (`width_`, `d_src_`).
- **Header Guards**: Canonical format `#ifndef CUDA_NPP_PROJECT_<FILE>_H_`.
- **Memory Safety & RAII**: Zero raw pointer leaks; device buffers are wrapped in RAII classes with non-copyable semantic enforcement.
- **Namespaces**: All components are organized under `namespace cuda_image_proc`.

---

## 8. License

This project is released under the [Apache 2.0 License](LICENSE).
