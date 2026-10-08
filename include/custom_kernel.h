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

#ifndef CUDA_NPP_PROJECT_CUSTOM_KERNEL_H_
#define CUDA_NPP_PROJECT_CUSTOM_KERNEL_H_

#include <string>
#include "common.h"
#include "image_io.h"

namespace cuda_image_proc {

// Custom CUDA Kernel Pipeline for image filtering and enhancement.
class CustomKernelProcessor {
 public:
  CustomKernelProcessor();
  ~CustomKernelProcessor();

  CustomKernelProcessor(const CustomKernelProcessor&) = delete;
  CustomKernelProcessor& operator=(const CustomKernelProcessor&) = delete;

  // Initializes device context.
  bool Initialize(int device_id = 0);

  // Executes shared-memory 2D Laplacian edge detection kernel.
  bool ApplyLaplacian(const ImageBuffer& input, ImageBuffer* output,
                      PerformanceMetrics* metrics = nullptr);

  // Executes shared-memory 2D sharpening kernel (unsharp mask).
  bool ApplySharpen(const ImageBuffer& input, ImageBuffer* output,
                    float strength = 1.5f,
                    PerformanceMetrics* metrics = nullptr);

  void Cleanup();

 private:
  bool is_initialized_;
  int device_id_;

#if defined(__CUDACC__) || defined(ENABLE_CUDA)
  uint8_t* d_src_;
  uint8_t* d_dst_;
  size_t allocated_size_;
  bool EnsureBuffers(size_t required_bytes);
#endif
};

}  // namespace cuda_image_proc

#endif  // CUDA_NPP_PROJECT_CUSTOM_KERNEL_H_
