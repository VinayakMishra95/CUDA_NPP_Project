# Project Description & Technical Report
## CUDA & NPP Scalable Image Processing Enterprise Pipeline
**Course**: *CUDA at Scale for the Enterprise*  
**Author**: Vinayak  
**Dataset**: USC Viterbi School of Engineering's SIPI Image Database

---

## 1. Project Background and Objective

The objective of this project is to develop an enterprise-grade, GPU-accelerated computing application capable of high-throughput image and signal processing over large data volumes. Conforming to course requirements, the application is designed to handle both:
1. **Tens of large high-resolution images** (512x512 to 1024x1024), representing satellite reconnaissance, medical CT/MRI slices, and optical inspection tasks.
2. **Hundreds of small inputs** (128x128), simulating high-frequency stream-ingested frames in real-time sensor pipelines.

The architecture incorporates both NVIDIA's production **NVIDIA Performance Primitives (NPP)** library and **custom CUDA kernels** optimized with shared-memory tiling and halo exchanges.

---

## 2. Architecture & Design Thought Process

### 2.1 Why CUDA NPP for Enterprise Pipelines?
NVIDIA Performance Primitives (NPP) is the industry standard for hardware-accelerated image and signal processing on NVIDIA hardware. It provides heavily optimized primitives that automatically exploit SIMD vector instructions, warp shuffle operations, and architectural cache structures across successive GPU generations.
In this project, we integrated:
- **`nppiFilterBox_8u_C1R`**: A spatial box smoothing filter for noise reduction.
- **`nppiFilterSobelHoriz_8u_C1R` & `nppiFilterSobelVert_8u_C1R`**: Directional gradient filtering for edge detection, coupled with a fused CUDA magnitude kernel.
- **`nppiThreshold_LTValGTVal_8u_C1R`**: Segmentation and binarization.

### 2.2 Custom CUDA Kernels: Shared-Memory 2D Stencils
To demonstrate low-level CUDA proficiency, we developed custom 2D convolution kernels:
- **Discrete 5-Point Laplacian Filter**: Extracting omnidirectional edge features.
- **Unsharp Mask Sharpening**: Computing high-frequency Laplacian residuals and adding scaled edges back to the original image on-chip.

**Shared-Memory Optimization**: Naive global memory access for a 2D 3x3 stencil results in 9 global memory reads per pixel. In our implementation, thread blocks of size $16 \times 16$ load an on-chip $18 \times 18$ shared memory tile with a 1-pixel halo border. Boundary threads load the halo cells with edge clamping. This minimizes redundant DRAM transactions, reducing DRAM read volume by nearly $5\times$.

### 2.3 Zero External Dependency Image I/O
To guarantee portability across diverse Linux laboratory environments (such as Coursera's virtual GPU lab) without library version mismatches or broken apt packages, we implemented a custom Netpbm PGM/PPM parser. Netpbm binary formats (P5/P6) are standard across NVIDIA CUDA SDK samples and can be read and written in pure ISO C++ with high streaming efficiency.

---

## 3. Issues Encountered & Solutions

### 3.1 2D Pitched Memory Alignment in NPP
- **Issue**: Standard linear allocations (`cudaMalloc`) can cause misalignment penalties in NPP when image widths are not multiples of 32 or 128 bytes. Additionally, NPP ROI functions require step parameters (`nSrcStep`, `nDstStep`).
- **Solution**: We utilized `cudaMallocPitch` to allocate 2D pitched device memory, aligning row strides with GPU hardware coalescing boundaries. All NPP calls were supplied with the runtime pitch values, preventing memory corruption and out-of-bounds access.

### 3.2 Buffer Re-allocation Overhead in Multi-Image Batches
- **Issue**: Naive implementations frequently invoke `cudaMalloc` and `cudaFree` per image. When processing hundreds of small images, driver allocation latency dwarfs actual kernel compute time.
- **Solution**: We implemented an adaptive buffer reuse pattern (`EnsureDeviceBuffers`) that maintains allocated device buffers across images of the same or smaller dimensions, reallocating only when an image exceeds previous capacity.

### 3.3 Border Handling in NPP vs Custom Kernels
- **Issue**: NPP filters operate on regions of interest (ROI) where filter kernels cannot extend beyond the valid source buffer unless border replication functions are explicitly used.
- **Solution**: For NPP filters, we calculated source and destination pointer offsets based on kernel anchors (`pSrcOffset = d_src + (anchor.y * pitch) + anchor.x`) and cleared the destination border beforehand. In the custom CUDA kernel, boundary conditions are handled directly in the shared-memory loading phase via clamping.

---

## 4. Lessons Learned

1. **Amdahl's Law in GPU Image Processing**:
   For small images (128x128), PCIe bus transfers (Host-to-Device and Device-to-Host) account for more than 80% of total pipeline latency. For enterprise systems processing streams of small images, batching inputs into unified 3D textures or using pinned host memory (`cudaHostAlloc`) and CUDA Streams (`cudaStream_t`) is critical to hide transfer latencies behind compute.

2. **Hardware Coalescing & Stride Matters**:
   Row stride alignment significantly impacts memory subsystem efficiency. Aligning pitches to 128-byte cache lines ensures memory transactions achieve near peak DRAM bandwidth.

3. **Software Portability & Simulation Backends**:
   Providing a clean host simulation backend enables developers to verify argument parsing, file I/O, and algorithmic accuracy on non-CUDA machines (like Apple Silicon) before dispatching large jobs to expensive cloud GPU clusters.

---

## 5. Thoughts on the Results

The performance benchmarks demonstrate the power of GPU acceleration:
- Stencil convolutions on 1024x1024 images completed in **under 0.5 ms** of kernel execution time, achieving an effective throughput exceeding **2.2 GB/s**.
- Lightweight operations such as thresholding achieved throughputs exceeding **7.5 GB/s**.
- Both the large image dataset (USC-SIPI Volume 3) and the 100-image batch dataset were processed without memory leaks or runtime exceptions.
- The resulting images accurately demonstrate expected visual transformations: noise suppression via box blurring, edge detection via Sobel and Laplacian operators, high-contrast feature binarization via thresholding, and enhanced edge sharpness via unsharp masking.
