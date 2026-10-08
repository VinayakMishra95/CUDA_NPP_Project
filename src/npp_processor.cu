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

#include "npp_processor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#if defined(__CUDACC__) || defined(ENABLE_CUDA)
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi_filtering_functions.h>
#include <nppi_threshold_and_compare_operations.h>

namespace cuda_image_proc {

namespace {

// Kernel to combine horizontal and vertical Sobel gradients into magnitude.
__global__ void SobelMagnitudeKernel(const uint8_t* __restrict__ horiz,
                                     int horiz_pitch,
                                     const uint8_t* __restrict__ vert,
                                     int vert_pitch,
                                     uint8_t* __restrict__ dst,
                                     int dst_pitch,
                                     int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    float gx = static_cast<float>(horiz[y * horiz_pitch + x]);
    float gy = static_cast<float>(vert[y * vert_pitch + x]);
    float mag = std::sqrt(gx * gx + gy * gy);
    dst[y * dst_pitch + x] = static_cast<uint8_t>(fminf(255.0f, mag));
  }
}

}  // namespace

NppImageProcessor::NppImageProcessor()
    : is_initialized_(false),
      device_id_(0),
      device_name_("None"),
      cc_major_(0),
      cc_minor_(0),
      total_global_mem_(0),
      d_src_(nullptr),
      d_dst_(nullptr),
      d_temp_(nullptr),
      src_pitch_(0),
      dst_pitch_(0),
      temp_pitch_(0),
      allocated_width_(0),
      allocated_height_(0) {}

NppImageProcessor::~NppImageProcessor() {
  Cleanup();
}

bool NppImageProcessor::Initialize(int device_id) {
  device_id_ = device_id;
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::cerr << "Error: No CUDA-capable devices detected.\n";
    return false;
  }

  CUDA_CHECK(cudaSetDevice(device_id_));
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, device_id_));

  device_name_ = props.name;
  cc_major_ = props.major;
  cc_minor_ = props.minor;
  total_global_mem_ = props.totalGlobalMem;

  std::cout << "NPP Processor initialized on GPU " << device_id_ << ": "
            << device_name_ << " (Compute Capability " << cc_major_ << "."
            << cc_minor_ << ", Global Memory: "
            << (total_global_mem_ / (1024 * 1024)) << " MB)\n";

  is_initialized_ = true;
  return true;
}

bool NppImageProcessor::EnsureDeviceBuffers(int width, int height) {
  if (allocated_width_ >= width && allocated_height_ >= height &&
      d_src_ != nullptr && d_dst_ != nullptr && d_temp_ != nullptr) {
    return true;
  }

  Cleanup();

  size_t pitch = 0;
  CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_src_), &pitch,
                             width * sizeof(uint8_t), height));
  src_pitch_ = static_cast<int>(pitch);

  CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_dst_), &pitch,
                             width * sizeof(uint8_t), height));
  dst_pitch_ = static_cast<int>(pitch);

  CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&d_temp_), &pitch,
                             width * sizeof(uint8_t), height));
  temp_pitch_ = static_cast<int>(pitch);

  allocated_width_ = width;
  allocated_height_ = height;
  return true;
}

bool NppImageProcessor::ApplyBoxFilter(const ImageBuffer& input,
                                      ImageBuffer* output,
                                      int kernel_size,
                                      PerformanceMetrics* metrics) {
  if (!is_initialized_ && !Initialize(0)) return false;
  if (input.empty()) return false;

  int width = input.width();
  int height = input.height();
  output->Resize(width, height, 1);
  EnsureDeviceBuffers(width, height);

  GpuTimer h2d_timer;
  GpuTimer kernel_timer;
  GpuTimer d2h_timer;

  // Host to Device Transfer
  h2d_timer.Start();
  CUDA_CHECK(cudaMemcpy2D(d_src_, src_pitch_, input.data(),
                          width * sizeof(uint8_t), width * sizeof(uint8_t),
                          height, cudaMemcpyHostToDevice));
  h2d_timer.Stop();

  // NPP Box Filter Configuration
  NppiSize oMaskSize = {kernel_size, kernel_size};
  NppiPoint oAnchor = {kernel_size / 2, kernel_size / 2};
  NppiSize oSizeROI = {width - (kernel_size - 1), height - (kernel_size - 1)};

  // Clear destination buffer to handle borders
  CUDA_CHECK(cudaMemset2D(d_dst_, dst_pitch_, 0, width, height));

  kernel_timer.Start();
  const Npp8u* pSrcOffset = d_src_ + (oAnchor.y * src_pitch_) + oAnchor.x;
  Npp8u* pDstOffset = d_dst_ + (oAnchor.y * dst_pitch_) + oAnchor.x;

  NPP_CHECK(nppiFilterBox_8u_C1R(pSrcOffset, src_pitch_, pDstOffset,
                                dst_pitch_, oSizeROI, oMaskSize, oAnchor));
  kernel_timer.Stop();

  // Device to Host Transfer
  d2h_timer.Start();
  CUDA_CHECK(cudaMemcpy2D(output->mutable_data(), width * sizeof(uint8_t),
                          d_dst_, dst_pitch_, width * sizeof(uint8_t),
                          height, cudaMemcpyDeviceToHost));
  d2h_timer.Stop();

  if (metrics != nullptr) {
    metrics->width = width;
    metrics->height = height;
    metrics->filter_name = "npp_box_" + std::to_string(kernel_size) + "x" +
                           std::to_string(kernel_size);
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

bool NppImageProcessor::ApplySobelFilter(const ImageBuffer& input,
                                        ImageBuffer* output,
                                        PerformanceMetrics* metrics) {
  if (!is_initialized_ && !Initialize(0)) return false;
  if (input.empty()) return false;

  int width = input.width();
  int height = input.height();
  output->Resize(width, height, 1);
  EnsureDeviceBuffers(width, height);

  GpuTimer h2d_timer;
  GpuTimer kernel_timer;
  GpuTimer d2h_timer;

  // Host to Device Transfer
  h2d_timer.Start();
  CUDA_CHECK(cudaMemcpy2D(d_src_, src_pitch_, input.data(),
                          width * sizeof(uint8_t), width * sizeof(uint8_t),
                          height, cudaMemcpyHostToDevice));
  h2d_timer.Stop();

  NppiSize oSizeROI = {width - 2, height - 2};
  CUDA_CHECK(cudaMemset2D(d_dst_, dst_pitch_, 0, width, height));
  CUDA_CHECK(cudaMemset2D(d_temp_, temp_pitch_, 0, width, height));

  kernel_timer.Start();
  const Npp8u* pSrcOffset = d_src_ + (1 * src_pitch_) + 1;
  Npp8u* pDstOffset = d_dst_ + (1 * dst_pitch_) + 1;
  Npp8u* pTempOffset = d_temp_ + (1 * temp_pitch_) + 1;

  // Horizontal Sobel into d_dst_
  NPP_CHECK(nppiFilterSobelHoriz_8u_C1R(pSrcOffset, src_pitch_, pDstOffset,
                                       dst_pitch_, oSizeROI));

  // Vertical Sobel into d_temp_
  NPP_CHECK(nppiFilterSobelVert_8u_C1R(pSrcOffset, src_pitch_, pTempOffset,
                                      temp_pitch_, oSizeROI));

  // Combine into gradient magnitude using CUDA kernel
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  SobelMagnitudeKernel<<<grid, block>>>(d_dst_, dst_pitch_, d_temp_,
                                        temp_pitch_, d_dst_, dst_pitch_,
                                        width, height);
  CUDA_CHECK(cudaGetLastError());
  kernel_timer.Stop();

  // Device to Host Transfer
  d2h_timer.Start();
  CUDA_CHECK(cudaMemcpy2D(output->mutable_data(), width * sizeof(uint8_t),
                          d_dst_, dst_pitch_, width * sizeof(uint8_t),
                          height, cudaMemcpyDeviceToHost));
  d2h_timer.Stop();

  if (metrics != nullptr) {
    metrics->width = width;
    metrics->height = height;
    metrics->filter_name = "npp_sobel";
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

bool NppImageProcessor::ApplyThreshold(const ImageBuffer& input,
                                      ImageBuffer* output,
                                      uint8_t threshold_val,
                                      PerformanceMetrics* metrics) {
  if (!is_initialized_ && !Initialize(0)) return false;
  if (input.empty()) return false;

  int width = input.width();
  int height = input.height();
  output->Resize(width, height, 1);
  EnsureDeviceBuffers(width, height);

  GpuTimer h2d_timer;
  GpuTimer kernel_timer;
  GpuTimer d2h_timer;

  // Host to Device Transfer
  h2d_timer.Start();
  CUDA_CHECK(cudaMemcpy2D(d_src_, src_pitch_, input.data(),
                          width * sizeof(uint8_t), width * sizeof(uint8_t),
                          height, cudaMemcpyHostToDevice));
  h2d_timer.Stop();

  NppiSize oSizeROI = {width, height};

  kernel_timer.Start();
  NPP_CHECK(nppiThreshold_LTValGTVal_8u_C1R(d_src_, src_pitch_, d_dst_,
                                           dst_pitch_, oSizeROI,
                                           threshold_val, 0,
                                           threshold_val, 255));
  kernel_timer.Stop();

  // Device to Host Transfer
  d2h_timer.Start();
  CUDA_CHECK(cudaMemcpy2D(output->mutable_data(), width * sizeof(uint8_t),
                          d_dst_, dst_pitch_, width * sizeof(uint8_t),
                          height, cudaMemcpyDeviceToHost));
  d2h_timer.Stop();

  if (metrics != nullptr) {
    metrics->width = width;
    metrics->height = height;
    metrics->filter_name = "npp_threshold_" + std::to_string(threshold_val);
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

void NppImageProcessor::Cleanup() {
  if (d_src_ != nullptr) {
    cudaFree(d_src_);
    d_src_ = nullptr;
  }
  if (d_dst_ != nullptr) {
    cudaFree(d_dst_);
    d_dst_ = nullptr;
  }
  if (d_temp_ != nullptr) {
    cudaFree(d_temp_);
    d_temp_ = nullptr;
  }
  allocated_width_ = 0;
  allocated_height_ = 0;
}

}  // namespace cuda_image_proc

#else

// ----------------------------------------------------------------------------
// CPU Simulation Fallback (for non-CUDA build / validation environments)
// ----------------------------------------------------------------------------

namespace cuda_image_proc {

NppImageProcessor::NppImageProcessor()
    : is_initialized_(false),
      device_id_(0),
      device_name_("Host CPU (Simulation)"),
      cc_major_(0),
      cc_minor_(0),
      total_global_mem_(0) {}

NppImageProcessor::~NppImageProcessor() {
  Cleanup();
}

bool NppImageProcessor::Initialize(int device_id) {
  device_id_ = device_id;
  device_name_ = "Host CPU (NPP Algorithmic Simulation)";
  cc_major_ = 0;
  cc_minor_ = 0;
  total_global_mem_ = 0;
  is_initialized_ = true;
  std::cout << "NPP Processor initialized in Host CPU Simulation mode.\n";
  return true;
}

bool NppImageProcessor::ApplyBoxFilter(const ImageBuffer& input,
                                      ImageBuffer* output,
                                      int kernel_size,
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

  int r = kernel_size / 2;
  float norm = 1.0f / static_cast<float>(kernel_size * kernel_size);

  for (int y = r; y < height - r; ++y) {
    for (int x = r; x < width - r; ++x) {
      int sum = 0;
      for (int ky = -r; ky <= r; ++ky) {
        for (int kx = -r; kx <= r; ++kx) {
          sum += src[(y + ky) * width + (x + kx)];
        }
      }
      dst[y * width + x] = static_cast<uint8_t>(std::round(sum * norm));
    }
  }

  double elapsed = timer.ElapsedMillis();
  if (metrics != nullptr) {
    metrics->width = width;
    metrics->height = height;
    metrics->filter_name = "sim_npp_box_" + std::to_string(kernel_size) + "x" +
                           std::to_string(kernel_size);
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

bool NppImageProcessor::ApplySobelFilter(const ImageBuffer& input,
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
      int gx = -src[(y - 1) * width + (x - 1)] + src[(y - 1) * width + (x + 1)] -
               2 * src[y * width + (x - 1)] + 2 * src[y * width + (x + 1)] -
               src[(y + 1) * width + (x - 1)] + src[(y + 1) * width + (x + 1)];

      int gy = -src[(y - 1) * width + (x - 1)] - 2 * src[(y - 1) * width + x] -
               src[(y - 1) * width + (x + 1)] + src[(y + 1) * width + (x - 1)] +
               2 * src[(y + 1) * width + x] + src[(y + 1) * width + (x + 1)];

      float mag = std::sqrt(static_cast<float>(gx * gx + gy * gy));
      dst[y * width + x] = static_cast<uint8_t>(std::min(255.0f, mag));
    }
  }

  double elapsed = timer.ElapsedMillis();
  if (metrics != nullptr) {
    metrics->width = width;
    metrics->height = height;
    metrics->filter_name = "sim_npp_sobel";
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

bool NppImageProcessor::ApplyThreshold(const ImageBuffer& input,
                                      ImageBuffer* output,
                                      uint8_t threshold_val,
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
  size_t total = static_cast<size_t>(width * height);

  for (size_t i = 0; i < total; ++i) {
    dst[i] = (src[i] >= threshold_val) ? 255 : 0;
  }

  double elapsed = timer.ElapsedMillis();
  if (metrics != nullptr) {
    metrics->width = width;
    metrics->height = height;
    metrics->filter_name = "sim_npp_threshold_" + std::to_string(threshold_val);
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

void NppImageProcessor::Cleanup() {}

}  // namespace cuda_image_proc

#endif  // defined(__CUDACC__) || defined(ENABLE_CUDA)
