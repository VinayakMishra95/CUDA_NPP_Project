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

#ifndef CUDA_NPP_PROJECT_NPP_PROCESSOR_H_
#define CUDA_NPP_PROJECT_NPP_PROCESSOR_H_

#include <string>
#include "common.h"
#include "image_io.h"

namespace cuda_image_proc {

// Wrapper class managing CUDA NPP image filtering pipelines.
class NppImageProcessor {
 public:
  NppImageProcessor();
  ~NppImageProcessor();

  // Non-copyable for RAII device resource safety
  NppImageProcessor(const NppImageProcessor&) = delete;
  NppImageProcessor& operator=(const NppImageProcessor&) = delete;

  // Initializes device resources and queries GPU capabilities.
  bool Initialize(int device_id = 0);

  // Executes Box Smoothing Filter on input image.
  bool ApplyBoxFilter(const ImageBuffer& input, ImageBuffer* output,
                      int kernel_size, PerformanceMetrics* metrics = nullptr);

  // Executes Sobel Edge Detection Filter on input image.
  bool ApplySobelFilter(const ImageBuffer& input, ImageBuffer* output,
                        PerformanceMetrics* metrics = nullptr);

  // Executes Threshold Binarization on input image.
  bool ApplyThreshold(const ImageBuffer& input, ImageBuffer* output,
                      uint8_t threshold_val,
                      PerformanceMetrics* metrics = nullptr);

  // Releases device allocations.
  void Cleanup();

  std::string GetDeviceName() const { return device_name_; }
  int GetComputeCapabilityMajor() const { return cc_major_; }
  int GetComputeCapabilityMinor() const { return cc_minor_; }
  size_t GetTotalGlobalMemory() const { return total_global_mem_; }

 private:
  bool is_initialized_;
  int device_id_;
  std::string device_name_;
  int cc_major_;
  int cc_minor_;
  size_t total_global_mem_;

#if defined(__CUDACC__) || defined(ENABLE_CUDA)
  uint8_t* d_src_;
  uint8_t* d_dst_;
  uint8_t* d_temp_;
  int src_pitch_;
  int dst_pitch_;
  int temp_pitch_;
  int allocated_width_;
  int allocated_height_;

  bool EnsureDeviceBuffers(int width, int height);
#endif
};

}  // namespace cuda_image_proc

#endif  // CUDA_NPP_PROJECT_NPP_PROCESSOR_H_
