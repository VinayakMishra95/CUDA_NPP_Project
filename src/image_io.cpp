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

#include "image_io.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace cuda_image_proc {

namespace {

// Skips whitespace and comment lines in Netpbm files.
void SkipWhitespaceAndComments(std::ifstream& infile) {
  while (true) {
    int ch = infile.peek();
    if (ch == EOF) {
      break;
    }
    if (std::isspace(ch)) {
      infile.get();
    } else if (ch == '#') {
      std::string comment_line;
      std::getline(infile, comment_line);
    } else {
      break;
    }
  }
}

}  // namespace

ImageBuffer::ImageBuffer() : width_(0), height_(0), channels_(1) {}

ImageBuffer::ImageBuffer(int width, int height, int channels)
    : width_(width), height_(height), channels_(channels),
      data_(static_cast<size_t>(width * height * channels), 0) {}

void ImageBuffer::Resize(int width, int height, int channels) {
  width_ = width;
  height_ = height;
  channels_ = channels;
  data_.assign(static_cast<size_t>(width * height * channels), 0);
}

void ImageBuffer::Clear() {
  width_ = 0;
  height_ = 0;
  channels_ = 1;
  data_.clear();
}

bool LoadImageFile(const std::string& filepath, ImageBuffer* image) {
  if (image == nullptr) {
    std::cerr << "Error: Null ImageBuffer pointer passed to LoadImageFile.\n";
    return false;
  }

  std::ifstream infile(filepath, std::ios::binary);
  if (!infile.is_open()) {
    std::cerr << "Error: Unable to open file for reading: " << filepath << "\n";
    return false;
  }

  std::string magic_number;
  infile >> magic_number;
  if (magic_number != "P5" && magic_number != "P6") {
    std::cerr << "Error: Unsupported format '" << magic_number
              << "' in " << filepath << " (Only P5 and P6 supported).\n";
    return false;
  }

  SkipWhitespaceAndComments(infile);
  int width = 0;
  infile >> width;

  SkipWhitespaceAndComments(infile);
  int height = 0;
  infile >> height;

  SkipWhitespaceAndComments(infile);
  int max_val = 0;
  infile >> max_val;

  if (width <= 0 || height <= 0 || max_val <= 0 || max_val > 255) {
    std::cerr << "Error: Invalid dimensions or max value in " << filepath
              << " (" << width << "x" << height << ", max=" << max_val << ")\n";
    return false;
  }

  // Consume single whitespace character after header before binary data
  infile.get();

  int channels = (magic_number == "P5") ? 1 : 3;
  image->Resize(width, height, channels);

  size_t total_bytes = static_cast<size_t>(width * height * channels);
  infile.read(reinterpret_cast<char*>(image->mutable_data()), total_bytes);

  if (!infile) {
    std::cerr << "Error: Unexpected EOF while reading image data from "
              << filepath << "\n";
    return false;
  }

  return true;
}

bool SaveImageFile(const std::string& filepath, const ImageBuffer& image) {
  if (image.empty() || image.width() <= 0 || image.height() <= 0) {
    std::cerr << "Error: Cannot save empty image to " << filepath << "\n";
    return false;
  }

  std::ofstream outfile(filepath, std::ios::binary);
  if (!outfile.is_open()) {
    std::cerr << "Error: Unable to open file for writing: " << filepath << "\n";
    return false;
  }

  const std::string magic_number = (image.channels() == 1) ? "P5" : "P6";
  outfile << magic_number << "\n";
  outfile << "# Created by CUDA Image Processing Enterprise Pipeline\n";
  outfile << image.width() << " " << image.height() << "\n";
  outfile << "255\n";

  outfile.write(reinterpret_cast<const char*>(image.data()), image.size());
  if (!outfile) {
    std::cerr << "Error: Failed to write image payload to " << filepath << "\n";
    return false;
  }

  return true;
}

bool ConvertRgbToGrayscale(const ImageBuffer& color_image,
                           ImageBuffer* gray_image) {
  if (color_image.channels() != 3 || gray_image == nullptr) {
    return false;
  }

  int width = color_image.width();
  int height = color_image.height();
  gray_image->Resize(width, height, 1);

  const uint8_t* src = color_image.data();
  uint8_t* dst = gray_image->mutable_data();

  size_t num_pixels = static_cast<size_t>(width * height);
  for (size_t i = 0; i < num_pixels; ++i) {
    float r = static_cast<float>(src[3 * i + 0]);
    float g = static_cast<float>(src[3 * i + 1]);
    float b = static_cast<float>(src[3 * i + 2]);
    // Standard ITU-R BT.601 conversion formula
    float gray = 0.299f * r + 0.587f * g + 0.114f * b;
    dst[i] = static_cast<uint8_t>(std::round(gray));
  }

  return true;
}

}  // namespace cuda_image_proc
