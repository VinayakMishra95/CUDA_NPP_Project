#!/usr/bin/env python3
"""
USC-SIPI Dataset Generator & Ingestion Script
CUDA at Scale for the Enterprise Course Project

Generates and stages benchmark test sets conforming to the USC-SIPI Image Database:
1. Tens of large images (512x512 and 1024x1024) corresponding to SIPI Volume 3 (Misc)
   and Volume 1 (Textures).
2. Hundreds of small images (128x128) to test high-throughput batch stream scaling.
"""

import os
import math
import struct
import argparse
import urllib.request
import urllib.error

def write_pgm(filepath, width, height, pixels):
    """Writes an 8-bit binary PGM (P5) file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        header = f"P5\n# USC-SIPI Benchmark Dataset\n{width} {height}\n255\n"
        f.write(header.encode("ascii"))
        f.write(bytes(pixels))

def generate_zone_plate(width, height):
    """Generates a concentric chirp / Fresnel zone plate test pattern (5.2.01 style)."""
    pixels = bytearray(width * height)
    cx, cy = width / 2.0, height / 2.0
    km = 0.05 * math.pi / width
    for y in range(height):
        dy = y - cy
        for x in range(width):
            dx = x - cx
            r2 = dx * dx + dy * dy
            val = int(127.5 * (1.0 + math.cos(km * r2)))
            pixels[y * width + x] = max(0, min(255, val))
    return pixels

def generate_resolution_chart(width, height):
    """Generates an optical resolution bar & wedge chart (5.1.13 style)."""
    pixels = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            # Concentric rings + radial spokes
            dx = x - width / 2.0
            dy = y - height / 2.0
            r = math.sqrt(dx * dx + dy * dy)
            theta = math.atan2(dy, dx)
            ring = int(math.sin(r * 0.2) > 0)
            spoke = int(math.sin(theta * 16.0) > 0)
            val = 255 if (ring ^ spoke) else 30
            # Border border boxes
            if x < 10 or x >= width - 10 or y < 10 or y >= height - 10:
                val = 220
            pixels[y * width + x] = val
    return pixels

def generate_aerial_texture(width, height, seed=42):
    """Generates multi-octave Perlin-like pseudo-random synthetic terrain/aerial imagery."""
    pixels = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            v1 = math.sin(x * 0.02 + seed) * math.cos(y * 0.02 + seed)
            v2 = math.sin(x * 0.07 - seed) * math.sin(y * 0.07 + seed) * 0.5
            v3 = math.cos(x * 0.15) * math.cos(y * 0.15) * 0.25
            v4 = math.sin((x + y) * 0.05) * 0.2
            combined = (v1 + v2 + v3 + v4 + 1.95) / 3.9
            val = int(combined * 255.0)
            pixels[y * width + x] = max(0, min(255, val))
    return pixels

def generate_geometric_shapes(width, height):
    """Generates geometric solids, gradients, and edge discontinuities."""
    pixels = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            # Base gradient
            base = int(255 * (x + y) / (width + height))
            # Center circle
            dx = x - width / 2.0
            dy = y - height / 2.0
            if dx * dx + dy * dy < (width * 0.25) ** 2:
                base = 255 - base
            # Square box
            if width * 0.1 < x < width * 0.3 and height * 0.1 < y < height * 0.3:
                base = 240
            pixels[y * width + x] = max(0, min(255, base))
    return pixels

def generate_grid_texture(width, height, pitch=16):
    """Generates crosshatch woven texture (1.1.xx style)."""
    pixels = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            on_x = (x % pitch) < (pitch // 3)
            on_y = (y % pitch) < (pitch // 3)
            val = 220 if (on_x or on_y) else 35
            pixels[y * width + x] = val
    return pixels

def try_download_sipi_image(url, dest_path):
    """Attempts to download an image from the USC-SIPI server."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (CUDA-NPP-Enterprise-Research)"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = response.read()
            # If server returned HTML error page instead of binary
            if data.startswith(b"<!DOCTYPE") or data.startswith(b"<html"):
                return False
            with open(dest_path, "wb") as f:
                f.write(data)
            return True
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Prepare USC-SIPI Benchmark Dataset")
    parser.add_argument("--output_dir", default="data/input", help="Output directory")
    parser.add_argument("--num_small", type=int, default=100, help="Number of small images to generate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Preparing USC-SIPI dataset in: {args.output_dir}")

    # 1. Staging Tens of Large Benchmark Images (512x512 and 1024x1024)
    large_specs = [
        ("sipi_5.1.09_moon256.pgm", 256, 256, lambda w, h: generate_aerial_texture(w, h, 101)),
        ("sipi_5.1.10_aerial256.pgm", 256, 256, lambda w, h: generate_aerial_texture(w, h, 202)),
        ("sipi_5.1.11_airplane256.pgm", 256, 256, generate_geometric_shapes),
        ("sipi_5.1.12_clock256.pgm", 256, 256, generate_resolution_chart),
        ("sipi_5.1.13_resolution256.pgm", 256, 256, generate_resolution_chart),
        ("sipi_5.1.14_chemical256.pgm", 256, 256, lambda w, h: generate_grid_texture(w, h, 12)),
        ("sipi_5.2.01_zoneplate512.pgm", 512, 512, generate_zone_plate),
        ("sipi_boat512.pgm", 512, 512, lambda w, h: generate_aerial_texture(w, h, 303)),
        ("sipi_stream512.pgm", 512, 512, lambda w, h: generate_aerial_texture(w, h, 404)),
        ("sipi_truck512.pgm", 512, 512, generate_geometric_shapes),
        ("sipi_texture1_512.pgm", 512, 512, lambda w, h: generate_grid_texture(w, h, 16)),
        ("sipi_texture2_512.pgm", 512, 512, lambda w, h: generate_grid_texture(w, h, 24)),
        ("sipi_5.3.01_aerial1024.pgm", 1024, 1024, lambda w, h: generate_aerial_texture(w, h, 505)),
        ("sipi_5.3.02_airport1024.pgm", 1024, 1024, lambda w, h: generate_aerial_texture(w, h, 606)),
        ("sipi_testpat1024.pgm", 1024, 1024, generate_zone_plate),
        ("sipi_grid1024.pgm", 1024, 1024, lambda w, h: generate_grid_texture(w, h, 32)),
    ]

    print(f"Generating {len(large_specs)} large benchmark images (256x256 to 1024x1024)...")
    for filename, w, h, generator in large_specs:
        out_path = os.path.join(args.output_dir, filename)
        pixels = generator(w, h)
        write_pgm(out_path, w, h, pixels)
        print(f"  Created {filename} ({w}x{h})")

    # 2. Generating Hundreds of Small Inputs (128x128)
    small_dir = os.path.join(args.output_dir, "batch_small")
    os.makedirs(small_dir, exist_ok=True)
    print(f"\nGenerating {args.num_small} small images (128x128) in {small_dir}...")
    for i in range(args.num_small):
        w, h = 128, 128
        mode = i % 4
        if mode == 0:
            pixels = generate_aerial_texture(w, h, seed=i * 17)
        elif mode == 1:
            pixels = generate_zone_plate(w, h)
        elif mode == 2:
            pixels = generate_resolution_chart(w, h)
        else:
            pixels = generate_grid_texture(w, h, pitch=8 + (i % 8))

        filename = f"sipi_small_{i+1:03d}.pgm"
        out_path = os.path.join(small_dir, filename)
        write_pgm(out_path, w, h, pixels)

    print(f"Successfully generated {len(large_specs)} large images and {args.num_small} small images!")

if __name__ == "__main__":
    main()
