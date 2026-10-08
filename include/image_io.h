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

#ifndef CUDA_NPP_PROJECT_IMAGE_IO_H_
#define CUDA_NPP_PROJECT_IMAGE_IO_H_

#include <cstdint>
#include <string>
#include <vector>

namespace cuda_image_proc {

// In-memory representation of an 8-bit image.
class ImageBuffer {
 public:
  ImageBuffer();
  ImageBuffer(int width, int height, int channels = 1);
  ~ImageBuffer() = default;

  // Rule of five compliance
  ImageBuffer(const ImageBuffer& other) = default;
  ImageBuffer& operator=(const ImageBuffer& other) = default;
  ImageBuffer(ImageBuffer&& other) noexcept = default;
  ImageBuffer& operator=(ImageBuffer&& other) noexcept = default;

  // Accessors
  int width() const { return width_; }
  int height() const { return height_; }
  int channels() const { return channels_; }
  size_t size() const { return data_.size(); }
  bool empty() const { return data_.empty(); }

  uint8_t* mutable_data() { return data_.data(); }
  const uint8_t* data() const { return data_.data(); }

  // Allocates buffer for width * height * channels.
  void Resize(int width, int height, int channels = 1);

  // Clears image contents.
  void Clear();

 private:
  int width_;
  int height_;
  int channels_;
  std::vector<uint8_t> data_;
};

// Reads a PGM (P5 grayscale) or PPM (P6 color) image from disk.
bool LoadImageFile(const std::string& filepath, ImageBuffer* image);

// Writes a PGM (P5 grayscale) or PPM (P6 color) image to disk.
bool SaveImageFile(const std::string& filepath, const ImageBuffer& image);

// Utility to convert color PPM (P6) to grayscale PGM (P5) using ITU-R BT.601 weights.
bool ConvertRgbToGrayscale(const ImageBuffer& color_image, ImageBuffer* gray_image);

}  // namespace cuda_image_proc

#endif  // CUDA_NPP_PROJECT_IMAGE_IO_H_
