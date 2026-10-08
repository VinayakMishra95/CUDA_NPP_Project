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

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "custom_kernel.h"
#include "image_io.h"
#include "npp_processor.h"

namespace cuda_image_proc {

struct CommandLineOptions {
  std::string input_dir = "data/input";
  std::string output_dir = "output";
  FilterType filter = FilterType::kAll;
  int threshold_val = 128;
  int kernel_size = 3;
  bool benchmark = false;
  bool verbose = false;
  std::string single_input_file = "";
  int device_id = 0;
};

void PrintUsage(const char* prog_name) {
  std::cout << "=================================================================\n";
  std::cout << "  CUDA & NPP Scalable Image Processing Enterprise Pipeline\n";
  std::cout << "=================================================================\n";
  std::cout << "Usage: " << prog_name << " [options]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --input_dir <path>       Directory with input PGM/PPM images (default: data/input)\n";
  std::cout << "  --output_dir <path>      Directory for processed output images (default: output)\n";
  std::cout << "  --input_file <path>      Process a specific single image instead of directory\n";
  std::cout << "  --filter <name>          Filter operation to apply:\n";
  std::cout << "                           'box'              - NPP 2D Box Blur\n";
  std::cout << "                           'sobel'            - NPP 2D Sobel Edge Detection\n";
  std::cout << "                           'threshold'        - NPP Binarization Thresholding\n";
  std::cout << "                           'custom_laplacian' - Custom CUDA Shared-Mem Laplacian\n";
  std::cout << "                           'custom_sharpen'   - Custom CUDA Unsharp Sharpening\n";
  std::cout << "                           'all'              - Execute full pipeline (default)\n";
  std::cout << "  --threshold_val <0-255>  Cutoff for threshold filter (default: 128)\n";
  std::cout << "  --kernel_size <size>     Odd integer mask size for box filter (default: 3)\n";
  std::cout << "  --benchmark              Enable detailed timing and throughput logging\n";
  std::cout << "  --verbose                Enable verbose diagnostic output\n";
  std::cout << "  --device <id>            CUDA device ID (default: 0)\n";
  std::cout << "  --help                   Display this help message\n\n";
  std::cout << "Example:\n";
  std::cout << "  " << prog_name << " --input_dir data/input --output_dir output --filter all --benchmark\n";
  std::cout << "=================================================================\n";
}

bool ParseCommandLine(int argc, char* argv[], CommandLineOptions* options) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return false;
    } else if (arg == "--input_dir" && i + 1 < argc) {
      options->input_dir = argv[++i];
    } else if (arg == "--output_dir" && i + 1 < argc) {
      options->output_dir = argv[++i];
    } else if (arg == "--input_file" && i + 1 < argc) {
      options->single_input_file = argv[++i];
    } else if (arg == "--filter" && i + 1 < argc) {
      options->filter = StringToFilterType(argv[++i]);
      if (options->filter == FilterType::kNone) {
        std::cerr << "Error: Unknown filter type: " << argv[i] << "\n";
        return false;
      }
    } else if (arg == "--threshold_val" && i + 1 < argc) {
      options->threshold_val = std::stoi(argv[++i]);
    } else if (arg == "--kernel_size" && i + 1 < argc) {
      options->kernel_size = std::stoi(argv[++i]);
    } else if (arg == "--device" && i + 1 < argc) {
      options->device_id = std::stoi(argv[++i]);
    } else if (arg == "--benchmark") {
      options->benchmark = true;
    } else if (arg == "--verbose") {
      options->verbose = true;
    } else {
      std::cerr << "Warning: Unrecognized argument '" << arg << "'\n";
    }
  }
  return true;
}

std::vector<std::string> GetFilesWithExtension(const std::string& directory,
                                               const std::string& ext) {
  std::vector<std::string> filepaths;
  DIR* dir = opendir(directory.c_str());
  if (dir == nullptr) {
    return filepaths;
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string filename = entry->d_name;
    if (filename.length() >= ext.length() &&
        filename.compare(filename.length() - ext.length(), ext.length(), ext) == 0) {
      std::string fullpath = directory + "/" + filename;
      filepaths.push_back(fullpath);
    }
  }
  closedir(dir);
  std::sort(filepaths.begin(), filepaths.end());
  return filepaths;
}

std::string GetBaseFilename(const std::string& filepath) {
  size_t last_slash = filepath.find_last_of("/\\");
  std::string name = (last_slash == std::string::npos)
                         ? filepath
                         : filepath.substr(last_slash + 1);
  size_t last_dot = name.find_last_of(".");
  return (last_dot == std::string::npos) ? name : name.substr(0, last_dot);
}

void PrintMetricsTable(const std::vector<PerformanceMetrics>& metrics_list) {
  std::cout << "\n";
  std::cout << "========================================================================================================\n";
  std::cout << "                                PERFORMANCE & BENCHMARK SUMMARY TABLE\n";
  std::cout << "========================================================================================================\n";
  std::cout << std::left
            << std::setw(28) << "Image File"
            << std::setw(12) << "Resolution"
            << std::setw(18) << "Filter"
            << std::right
            << std::setw(11) << "H2D (ms)"
            << std::setw(14) << "Kernel (ms)"
            << std::setw(11) << "D2H (ms)"
            << std::setw(13) << "Total (ms)"
            << std::setw(15) << "Throughput(MB/s)"
            << "\n";
  std::cout << "--------------------------------------------------------------------------------------------------------\n";

  double total_time_acc = 0.0;
  double total_kernel_acc = 0.0;
  size_t total_bytes_processed = 0;

  for (const auto& m : metrics_list) {
    std::string res = std::to_string(m.width) + "x" + std::to_string(m.height);
    std::cout << std::left
              << std::setw(28) << (m.filename.length() > 27 ? m.filename.substr(0, 24) + "..." : m.filename)
              << std::setw(12) << res
              << std::setw(18) << m.filter_name
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(11) << m.h2d_time_ms
              << std::setw(14) << m.kernel_time_ms
              << std::setw(11) << m.d2h_time_ms
              << std::setw(13) << m.total_time_ms
              << std::setprecision(1)
              << std::setw(15) << m.throughput_mbytes_per_sec
              << "\n";

    total_time_acc += m.total_time_ms;
    total_kernel_acc += m.kernel_time_ms;
    total_bytes_processed += static_cast<size_t>(m.width * m.height);
  }

  std::cout << "========================================================================================================\n";
  double total_mb = static_cast<double>(total_bytes_processed) / (1024.0 * 1024.0);
  std::cout << "Total Operations:       " << metrics_list.size() << "\n";
  std::cout << "Total Data Processed:   " << std::fixed << std::setprecision(2) << total_mb << " MB\n";
  std::cout << "Total Wall-Clock Time:  " << std::fixed << std::setprecision(2) << total_time_acc << " ms\n";
  std::cout << "Total Pure Kernel Time: " << std::fixed << std::setprecision(2) << total_kernel_acc << " ms\n";
  if (total_time_acc > 0.0) {
    double avg_throughput = total_mb / (total_time_acc / 1000.0);
    std::cout << "Average Throughput:     " << std::fixed << std::setprecision(2) << avg_throughput << " MB/s\n";
  }
  std::cout << "========================================================================================================\n\n";
}

}  // namespace cuda_image_proc

int main(int argc, char* argv[]) {
  using namespace cuda_image_proc;

  CommandLineOptions options;
  if (!ParseCommandLine(argc, argv, &options)) {
    return 0;
  }

  // Ensure output directory exists
  mkdir(options.output_dir.c_str(), 0777);

  std::vector<std::string> input_files;
  if (!options.single_input_file.empty()) {
    input_files.push_back(options.single_input_file);
  } else {
    auto pgm_files = GetFilesWithExtension(options.input_dir, ".pgm");
    auto ppm_files = GetFilesWithExtension(options.input_dir, ".ppm");
    input_files.insert(input_files.end(), pgm_files.begin(), pgm_files.end());
    input_files.insert(input_files.end(), ppm_files.begin(), ppm_files.end());
  }

  if (input_files.empty()) {
    std::cerr << "Error: No .pgm or .ppm images found in input location: "
              << options.input_dir << "\n";
    std::cerr << "Hint: Run scripts/prepare_dataset.py or supply --input_file <path>\n";
    return 1;
  }

  std::cout << "Initializing CUDA / NPP Image Processing Engine...\n";
  std::cout << "Input dataset contains " << input_files.size() << " image(s).\n";
  std::cout << "Target output directory: " << options.output_dir << "\n";

  NppImageProcessor npp_processor;
  npp_processor.Initialize(options.device_id);

  CustomKernelProcessor custom_processor;
  custom_processor.Initialize(options.device_id);

  std::vector<PerformanceMetrics> all_metrics;

  for (size_t idx = 0; idx < input_files.size(); ++idx) {
    const std::string& input_path = input_files[idx];
    std::string base_name = GetBaseFilename(input_path);

    ImageBuffer raw_image;
    if (!LoadImageFile(input_path, &raw_image)) {
      std::cerr << "Failed to load " << input_path << ", skipping.\n";
      continue;
    }

    ImageBuffer gray_image;
    if (raw_image.channels() == 3) {
      ConvertRgbToGrayscale(raw_image, &gray_image);
    } else {
      gray_image = raw_image;
    }

    std::cout << "[" << (idx + 1) << "/" << input_files.size() << "] Processing: "
              << base_name << " (" << gray_image.width() << "x" << gray_image.height() << ")\n";

    // NPP Box Filter
    if (options.filter == FilterType::kAll || options.filter == FilterType::kBoxFilter) {
      ImageBuffer out_box;
      PerformanceMetrics m;
      m.filename = base_name;
      if (npp_processor.ApplyBoxFilter(gray_image, &out_box, options.kernel_size, &m)) {
        std::string out_path = options.output_dir + "/" + base_name + "_npp_box.pgm";
        SaveImageFile(out_path, out_box);
        all_metrics.push_back(m);
      }
    }

    // NPP Sobel Filter
    if (options.filter == FilterType::kAll || options.filter == FilterType::kSobelFilter) {
      ImageBuffer out_sobel;
      PerformanceMetrics m;
      m.filename = base_name;
      if (npp_processor.ApplySobelFilter(gray_image, &out_sobel, &m)) {
        std::string out_path = options.output_dir + "/" + base_name + "_npp_sobel.pgm";
        SaveImageFile(out_path, out_sobel);
        all_metrics.push_back(m);
      }
    }

    // NPP Threshold
    if (options.filter == FilterType::kAll || options.filter == FilterType::kThreshold) {
      ImageBuffer out_thresh;
      PerformanceMetrics m;
      m.filename = base_name;
      if (npp_processor.ApplyThreshold(gray_image, &out_thresh,
                                       static_cast<uint8_t>(options.threshold_val), &m)) {
        std::string out_path = options.output_dir + "/" + base_name + "_npp_thresh.pgm";
        SaveImageFile(out_path, out_thresh);
        all_metrics.push_back(m);
      }
    }

    // Custom Shared-Memory Laplacian Kernel
    if (options.filter == FilterType::kAll || options.filter == FilterType::kCustomLaplacian) {
      ImageBuffer out_lap;
      PerformanceMetrics m;
      m.filename = base_name;
      if (custom_processor.ApplyLaplacian(gray_image, &out_lap, &m)) {
        std::string out_path = options.output_dir + "/" + base_name + "_custom_laplacian.pgm";
        SaveImageFile(out_path, out_lap);
        all_metrics.push_back(m);
      }
    }

    // Custom Shared-Memory Sharpening Kernel
    if (options.filter == FilterType::kAll || options.filter == FilterType::kCustomSharpen) {
      ImageBuffer out_sharp;
      PerformanceMetrics m;
      m.filename = base_name;
      if (custom_processor.ApplySharpen(gray_image, &out_sharp, 1.5f, &m)) {
        std::string out_path = options.output_dir + "/" + base_name + "_custom_sharpen.pgm";
        SaveImageFile(out_path, out_sharp);
        all_metrics.push_back(m);
      }
    }
  }

  if (options.benchmark || !all_metrics.empty()) {
    PrintMetricsTable(all_metrics);
  }

  std::cout << "All processing completed successfully.\n";
  return 0;
}
