// Copyright 2026 Vinayak. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "custom_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#if defined(__CUDACC__) || defined(ENABLE_CUDA)
#include <cuda_runtime.h>

namespace cuda_image_proc {

namespace {

constexpr int kTileDim = 16;
constexpr int kHaloDim = kTileDim + 2;

// 2D Laplacian Edge Detection Kernel with Shared Memory Tiling and Halo Exchange
__global__ void LaplacianSharedMemKernel(const uint8_t* __restrict__ src,
                                         uint8_t* __restrict__ dst,
                                         int width, int height) {
  __shared__ float s_tile[kHaloDim][kHaloDim];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int gx = blockIdx.x * kTileDim + tx;
  int gy = blockIdx.y * kTileDim + ty;

  // Load primary center pixel into shared memory
  int clamped_x = min(max(gx, 0), width - 1);
  int clamped_y = min(max(gy, 0), height - 1);
  s_tile[ty + 1][tx + 1] = static_cast<float>(src[clamped_y * width + clamped_x]);

  // Load halo borders
  // Left border
  if (tx == 0) {
    int left_x = min(max(gx - 1, 0), width - 1);
    s_tile[ty + 1][0] = static_cast<float>(src[clamped_y * width + left_x]);
  }
  // Right border
  if (tx == kTileDim - 1) {
    int right_x = min(max(gx + 1, 0), width - 1);
    s_tile[ty + 1][kHaloDim - 1] = static_cast<float>(src[clamped_y * width + right_x]);
  }
  // Top border
  if (ty == 0) {
    int top_y = min(max(gy - 1, 0), height - 1);
    s_tile[0][tx + 1] = static_cast<float>(src[top_y * width + clamped_x]);
  }
  // Bottom border
  if (ty == kTileDim - 1) {
    int bottom_y = min(max(gy + 1, 0), height - 1);
    s_tile[kHaloDim - 1][tx + 1] = static_cast<float>(src[bottom_y * width + clamped_x]);
  }

  // Corner halos
  if (tx == 0 && ty == 0) {
    s_tile[0][0] = static_cast<float>(
        src[min(max(gy - 1, 0), height - 1) * width + min(max(gx - 1, 0), width - 1)]);
  }
  if (tx == kTileDim - 1 && ty == 0) {
    s_tile[0][kHaloDim - 1] = static_cast<float>(
        src[min(max(gy - 1, 0), height - 1) * width + min(max(gx + 1, 0), width - 1)]);
  }
  if (tx == 0 && ty == kTileDim - 1) {
    s_tile[kHaloDim - 1][0] = static_cast<float>(
        src[min(max(gy + 1, 0), height - 1) * width + min(max(gx - 1, 0), width - 1)]);
  }
  if (tx == kTileDim - 1 && ty == kTileDim - 1) {
    s_tile[kHaloDim - 1][kHaloDim - 1] = static_cast<float>(
        src[min(max(gy + 1, 0), height - 1) * width + min(max(gx + 1, 0), width - 1)]);
  }

  __syncthreads();

  // Compute 5-point discrete 2D Laplacian operator
  if (gx < width && gy < height) {
    int sx = tx + 1;
    int sy = ty + 1;

    float lap = (s_tile[sy - 1][sx] + s_tile[sy + 1][sx] +
                 s_tile[sy][sx - 1] + s_tile[sy][sx + 1]) -
                4.0f * s_tile[sy][sx];

    float abs_lap = fabsf(lap);
    dst[gy * width + gx] = static_cast<uint8_t>(fminf(255.0f, abs_lap));
  }
}

// 2D Sharpening Kernel (Unsharp Masking via Laplacian)
__global__ void SharpenSharedMemKernel(const uint8_t* __restrict__ src,
                                       uint8_t* __restrict__ dst,
                                       int width, int height, float strength) {
  __shared__ float s_tile[kHaloDim][kHaloDim];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int gx = blockIdx.x * kTileDim + tx;
  int gy = blockIdx.y * kTileDim + ty;

  int clamped_x = min(max(gx, 0), width - 1);
  int clamped_y = min(max(gy, 0), height - 1);
  s_tile[ty + 1][tx + 1] = static_cast<float>(src[clamped_y * width + clamped_x]);

  if (tx == 0) {
    s_tile[ty + 1][0] = static_cast<float>(
        src[clamped_y * width + min(max(gx - 1, 0), width - 1)]);
  }
  if (tx == kTileDim - 1) {
    s_tile[ty + 1][kHaloDim - 1] = static_cast<float>(
        src[clamped_y * width + min(max(gx + 1, 0), width - 1)]);
  }
  if (ty == 0) {
    s_tile[0][tx + 1] = static_cast<float>(
        src[min(max(gy - 1, 0), height - 1) * width + clamped_x]);
  }
  if (ty == kTileDim - 1) {
    s_tile[kHaloDim - 1][tx + 1] = static_cast<float>(
        src[min(max(gy + 1, 0), height - 1) * width + clamped_x]);
  }

  __syncthreads();

  if (gx < width && gy < height) {
    int sx = tx + 1;
    int sy = ty + 1;

    float center = s_tile[sy][sx];
    float lap = (s_tile[sy - 1][sx] + s_tile[sy + 1][sx] +
                 s_tile[sy][sx - 1] + s_tile[sy][sx + 1]) -
                4.0f * center;

    // Sharpening formula: I_sharp = I_center - strength * Laplacian
    float sharp = center - strength * lap;
    dst[gy * width + gx] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, sharp)));
  }
}

}  // namespace

CustomKernelProcessor::CustomKernelProcessor()
    : is_initialized_(false),
      device_id_(0),
      d_src_(nullptr),
      d_dst_(nullptr),
      allocated_size_(0) {}

CustomKernelProcessor::~CustomKernelProcessor() {
  Cleanup();
}

bool CustomKernelProcessor::Initialize(int device_id) {
  device_id_ = device_id;
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::cerr << "Error: No CUDA-capable devices detected.\n";
    return false;
  }
  CUDA_CHECK(cudaSetDevice(device_id_));
  is_initialized_ = true;
  return true;
}

bool CustomKernelProcessor::EnsureBuffers(size_t required_bytes) {
  if (allocated_size_ >= required_bytes && d_src_ != nullptr && d_dst_ != nullptr) {
    return true;
  }
  Cleanup();
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_src_), required_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dst_), required_bytes));
  allocated_size_ = required_bytes;
  return true;
}

bool CustomKernelProcessor::ApplyLaplacian(const ImageBuffer& input,
                                           ImageBuffer* output,
                                           PerformanceMetrics* metrics) {
  if (!is_initialized_ && !Initialize(0)) return false;
  if (input.empty()) return false;

  int width = input.width();
  int height = input.height();
  size_t bytes = static_cast<size_t>(width * height);
  output->Resize(width, height, 1);
  EnsureBuffers(bytes);

  GpuTimer h2d_timer;
  GpuTimer kernel_timer;
  GpuTimer d2h_timer;

  h2d_timer.Start();
  CUDA_CHECK(cudaMemcpy(d_src_, input.data(), bytes, cudaMemcpyHostToDevice));
  h2d_timer.Stop();

  dim3 block(kTileDim, kTileDim);
  dim3 grid((width + kTileDim - 1) / kTileDim, (height + kTileDim - 1) / kTileDim);

  kernel_timer.Start();
  LaplacianSharedMemKernel<<<grid, block>>>(d_src_, d_dst_, width, height);
  CUDA_CHECK(cudaGetLastError());
  kernel_timer.Stop();

  d2h_timer.Start();
  CUDA_CHECK(cudaMemcpy(output->mutable_data(), d_dst_, bytes, cudaMemcpyDeviceToHost));
  d2h_timer.Stop();

  if (metrics != nullptr) {
    metrics->width = width;
    metrics->height = height;
    metrics->filter_name = "custom_laplacian";
    metrics->h2d_time_ms = h2d_timer.ElapsedMillis();
    metrics->kernel_time_ms = kernel_timer.ElapsedMillis();
    metrics->d2h_time_ms = d2h_timer.ElapsedMillis();
    metrics->total_time_ms = metrics->h2d_time_ms + metrics->kernel_time_ms +
                             metrics->d2h_time_ms;
    double mbytes = static_cast<double>(width * height) / (1024.0 * 1024.0);
    metrics->throughput_mbytes_per_sec =
        (metrics->total_time_ms > 0.0)
            ? (mbytes / (metrics->total_time_ms / 1000.0))
            : 0.0;
  }

  return true;
}

bool CustomKernelProcessor::ApplySharpen(const ImageBuffer& input,
                                         ImageBuffer* output,
                                         float strength,
                                         PerformanceMetrics* metrics) {
  if (!is_initialized_ && !Initialize(0)) return false;
  if (input.empty()) return false;

  int width = input.width();
  int height = input.height();
  size_t bytes = static_cast<size_t>(width * height);
  output->Resize(width, height, 1);
  EnsureBuffers(bytes);

  GpuTimer h2d_timer;
  GpuTimer kernel_timer;
  GpuTimer d2h_timer;

  h2d_timer.Start();
  CUDA_CHECK(cudaMemcpy(d_src_, input.data(), bytes, cudaMemcpyHostToDevice));
  h2d_timer.Stop();

  dim3 block(kTileDim, kTileDim);
  dim3 grid((width + kTileDim - 1) / kTileDim, (height + kTileDim - 1) / kTileDim);

  kernel_timer.Start();
  SharpenSharedMemKernel<<<grid, block>>>(d_src_, d_dst_, width, height, strength);
  CUDA_CHECK(cudaGetLastError());
  kernel_timer.Stop();

  d2h_timer.Start();
  CUDA_CHECK(cudaMemcpy(output->mutable_data(), d_dst_, bytes, cudaMemcpyDeviceToHost));
  d2h_timer.Stop();

  if (metrics != nullptr) {
    metrics->width = width;
    metrics->height = height;
    metrics->filter_name = "custom_sharpen";
    metrics->h2d_time_ms = h2d_timer.ElapsedMillis();
    metrics->kernel_time_ms = kernel_timer.ElapsedMillis();
    metrics->d2h_time_ms = d2h_timer.ElapsedMillis();
    metrics->total_time_ms = metrics->h2d_time_ms + metrics->kernel_time_ms +
                             metrics->d2h_time_ms;
    double mbytes = static_cast<double>(width * height) / (1024.0 * 1024.0);
    metrics->throughput_mbytes_per_sec =
        (metrics->total_time_ms > 0.0)
            ? (mbytes / (metrics->total_time_ms / 1000.0))
            : 0.0;
  }

  return true;
}

void CustomKernelProcessor::Cleanup() {
  if (d_src_ != nullptr) {
    cudaFree(d_src_);
    d_src_ = nullptr;
  }
  if (d_dst_ != nullptr) {
    cudaFree(d_dst_);
    d_dst_ = nullptr;
  }
  allocated_size_ = 0;
}

}  // namespace cuda_image_proc

#else

// ----------------------------------------------------------------------------
// CPU Simulation Fallback (for non-CUDA build / validation environments)
// ----------------------------------------------------------------------------

namespace cuda_image_proc {

CustomKernelProcessor::CustomKernelProcessor()
    : is_initialized_(false), device_id_(0) {}

CustomKernelProcessor::~CustomKernelProcessor() {
  Cleanup();
}

bool CustomKernelProcessor::Initialize(int device_id) {
  device_id_ = device_id;
  is_initialized_ = true;
  return true;
}

bool CustomKernelProcessor::ApplyLaplacian(const ImageBuffer& input,
                                           ImageBuffer* output,
                                           PerformanceMetrics* metrics) {
  if (!is_initialized_) Initialize(0);
  if (input.empty()) return false;

  int width = input.width();
  int height = input.height();
  output->Resize(width, height, 1);

  CpuTimer timer;
  timer.Start();

  const uint8_t* src = input.data();
  uint8_t* dst = output->mutable_data();
  std::memset(dst, 0, width * height);

  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      float center = src[y * width + x];
      float lap = (src[(y - 1) * width + x] + src[(y + 1) * width + x] +
                   src[y * width + (x - 1)] + src[y * width + (x + 1)]) -
                  4.0f * center;
      dst[y * width + x] = static_cast<uint8_t>(std::min(255.0f, std::abs(lap)));
    }
  }

  double elapsed = timer.ElapsedMillis();
  if (metrics != nullptr) {
    metrics->width = width;
    metrics->height = height;
    metrics->filter_name = "sim_custom_laplacian";
    metrics->h2d_time_ms = 0.05;
    metrics->kernel_time_ms = elapsed;
    metrics->d2h_time_ms = 0.05;
    metrics->total_time_ms = elapsed + 0.1;
    double mbytes = static_cast<double>(width * height) / (1024.0 * 1024.0);
    metrics->throughput_mbytes_per_sec =
        (metrics->total_time_ms > 0.0)
            ? (mbytes / (metrics->total_time_ms / 1000.0))
            : 0.0;
  }
  return true;
}

bool CustomKernelProcessor::ApplySharpen(const ImageBuffer& input,
                                         ImageBuffer* output,
                                         float strength,
                                         PerformanceMetrics* metrics) {
  if (!is_initialized_) Initialize(0);
  if (input.empty()) return false;

  int width = input.width();
  int height = input.height();
  output->Resize(width, height, 1);

  CpuTimer timer;
  timer.Start();

  const uint8_t* src = input.data();
  uint8_t* dst = output->mutable_data();
  std::memset(dst, 0, width * height);

  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      float center = src[y * width + x];
      float lap = (src[(y - 1) * width + x] + src[(y + 1) * width + x] +
                   src[y * width + (x - 1)] + src[y * width + (x + 1)]) -
                  4.0f * center;
      float sharp = center - strength * lap;
      dst[y * width + x] = static_cast<uint8_t>(
          std::min(255.0f, std::max(0.0f, sharp)));
    }
  }

  double elapsed = timer.ElapsedMillis();
  if (metrics != nullptr) {
    metrics->width = width;
    metrics->height = height;
    metrics->filter_name = "sim_custom_sharpen";
    metrics->h2d_time_ms = 0.05;
    metrics->kernel_time_ms = elapsed;
    metrics->d2h_time_ms = 0.05;
    metrics->total_time_ms = elapsed + 0.1;
    double mbytes = static_cast<double>(width * height) / (1024.0 * 1024.0);
    metrics->throughput_mbytes_per_sec =
        (metrics->total_time_ms > 0.0)
            ? (mbytes / (metrics->total_time_ms / 1000.0))
            : 0.0;
  }
  return true;
}

void CustomKernelProcessor::Cleanup() {}

}  // namespace cuda_image_proc

#endif  // defined(__CUDACC__) || defined(ENABLE_CUDA)
