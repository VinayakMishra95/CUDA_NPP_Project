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

#ifndef CUDA_NPP_PROJECT_COMMON_H_
#define CUDA_NPP_PROJECT_COMMON_H_

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#if defined(__CUDACC__) || defined(ENABLE_CUDA)
#include <cuda_runtime.h>
#include <npp.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__,        \
                   __LINE__, cudaGetErrorString(err), static_cast<int>(err)); \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

#define NPP_CHECK(call)                                                       \
  do {                                                                        \
    NppStatus status = (call);                                                \
    if (status != NPP_SUCCESS) {                                              \
      std::fprintf(stderr, "NPP error at %s:%d: status code %d\n", __FILE__,  \
                   __LINE__, static_cast<int>(status));                       \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

#else

// Fallback definitions for compilation and dry-run validation on hosts
// without an NVIDIA GPU or CUDA runtime.
typedef int cudaError_t;
constexpr int cudaSuccess = 0;
typedef int NppStatus;
constexpr int NPP_SUCCESS = 0;

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    int res = (call);                                                         \
    if (res != 0) {                                                           \
      std::fprintf(stderr, "Simulation error at %s:%d: %d\n", __FILE__,       \
                   __LINE__, res);                                            \
    }                                                                         \
  } while (0)

#define NPP_CHECK(call)                                                       \
  do {                                                                        \
    int res = (call);                                                         \
    if (res != 0) {                                                           \
      std::fprintf(stderr, "Simulation NPP error at %s:%d: %d\n", __FILE__,   \
                   __LINE__, res);                                            \
    }                                                                         \
  } while (0)

#endif  // defined(__CUDACC__) || defined(ENABLE_CUDA)

namespace cuda_image_proc {

// Filter types supported across CUDA NPP and Custom Kernel pipelines.
enum class FilterType {
  kNone,
  kBoxFilter,
  kSobelFilter,
  kThreshold,
  kCustomLaplacian,
  kCustomSharpen,
  kAll
};

// Converts FilterType enum to human-readable string.
inline std::string FilterTypeToString(FilterType filter) {
  switch (filter) {
    case FilterType::kNone:
      return "none";
    case FilterType::kBoxFilter:
      return "box";
    case FilterType::kSobelFilter:
      return "sobel";
    case FilterType::kThreshold:
      return "threshold";
    case FilterType::kCustomLaplacian:
      return "custom_laplacian";
    case FilterType::kCustomSharpen:
      return "custom_sharpen";
    case FilterType::kAll:
      return "all";
    default:
      return "unknown";
  }
}

// Parses string input into FilterType.
inline FilterType StringToFilterType(const std::string& str) {
  if (str == "box" || str == "boxfilter") return FilterType::kBoxFilter;
  if (str == "sobel" || str == "sobelfilter") return FilterType::kSobelFilter;
  if (str == "threshold" || str == "thresh") return FilterType::kThreshold;
  if (str == "custom_laplacian" || str == "laplacian") return FilterType::kCustomLaplacian;
  if (str == "custom_sharpen" || str == "sharpen") return FilterType::kCustomSharpen;
  if (str == "all") return FilterType::kAll;
  return FilterType::kNone;
}

// High-resolution CPU timer for host metrics.
class CpuTimer {
 public:
  CpuTimer() : start_time_(std::chrono::high_resolution_clock::now()) {}

  void Start() {
    start_time_ = std::chrono::high_resolution_clock::now();
  }

  double ElapsedMillis() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time_;
    return elapsed.count();
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

#if defined(__CUDACC__) || defined(ENABLE_CUDA)
// High-resolution GPU CUDA Event timer.
class GpuTimer {
 public:
  GpuTimer() {
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
  }

  ~GpuTimer() {
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
  }

  void Start(cudaStream_t stream = 0) {
    cudaEventRecord(start_event_, stream);
  }

  void Stop(cudaStream_t stream = 0) {
    cudaEventRecord(stop_event_, stream);
  }

  float ElapsedMillis() {
    cudaEventSynchronize(stop_event_);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start_event_, stop_event_);
    return milliseconds;
  }

 private:
  cudaEvent_t start_event_;
  cudaEvent_t stop_event_;
};
#endif  // defined(__CUDACC__) || defined(ENABLE_CUDA)

// Structure holding performance profile statistics.
struct PerformanceMetrics {
  std::string filename;
  int width = 0;
  int height = 0;
  std::string filter_name;
  double h2d_time_ms = 0.0;
  double kernel_time_ms = 0.0;
  double d2h_time_ms = 0.0;
  double total_time_ms = 0.0;
  double throughput_mbytes_per_sec = 0.0;
};

}  // namespace cuda_image_proc

#endif  // CUDA_NPP_PROJECT_COMMON_H_
