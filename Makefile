# ==============================================================================
# Makefile for CUDA & NPP Enterprise Image Processing Pipeline
# Course: CUDA at Scale for the Enterprise
# ==============================================================================

# CUDA Toolkit Configuration
CUDA_PATH ?= /usr/local/cuda
NVCC ?= $(CUDA_PATH)/bin/nvcc
HOST_CXX ?= g++

# GPU Architecture Capabilities (Volta, Turing, Ampere)
ARCH_FLAGS ?= -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86

# Compilation Flags
CXXFLAGS ?= -O3 -std=c++14 -Wall -Wextra
NVCCFLAGS ?= -O3 -std=c++14 $(ARCH_FLAGS) --compiler-options "-Wall -Wextra" -DENABLE_CUDA

# Include and Library Directories
INCLUDES := -Iinclude -I$(CUDA_PATH)/include
LIBS := -L$(CUDA_PATH)/lib64 -lcudart -lnppc -lnppif -lnppig -lnppitc

# Directories
SRC_DIR := src
INC_DIR := include
BUILD_DIR := build
BIN_DIR := bin
OUTPUT_DIR := output
DATA_DIR := data/input
ARTIFACTS_DIR := artifacts

TARGET := $(BIN_DIR)/cuda_image_proc
SIM_TARGET := $(BIN_DIR)/sim_image_proc

# Source files
CPP_SRCS := $(SRC_DIR)/image_io.cpp $(SRC_DIR)/main.cpp
CU_SRCS := $(SRC_DIR)/npp_processor.cu $(SRC_DIR)/custom_kernel.cu

# Object files
OBJS := $(BUILD_DIR)/image_io.o \
        $(BUILD_DIR)/main.o \
        $(BUILD_DIR)/npp_processor.o \
        $(BUILD_DIR)/custom_kernel.o

SIM_OBJS := $(BUILD_DIR)/sim_image_io.o \
            $(BUILD_DIR)/sim_main.o \
            $(BUILD_DIR)/sim_npp_processor.o \
            $(BUILD_DIR)/sim_custom_kernel.o

.PHONY: all clean run benchmark package help sim test

all: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	@echo "Linking CUDA Target: $@"
	$(NVCC) $(NVCCFLAGS) $(OBJS) $(LIBS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@echo "Compiling C++ Source: $<"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	@echo "Compiling CUDA Source: $<"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# CPU Simulation Target (Compiles with standard C++ compiler when NVCC is unavailable)
sim: $(SIM_TARGET)

$(SIM_TARGET): $(SIM_OBJS) | $(BIN_DIR)
	@echo "Linking CPU Simulation Target: $@"
	$(HOST_CXX) $(CXXFLAGS) $(SIM_OBJS) -o $@

$(BUILD_DIR)/sim_image_io.o: $(SRC_DIR)/image_io.cpp | $(BUILD_DIR)
	$(HOST_CXX) $(CXXFLAGS) -I$(INC_DIR) -c $< -o $@

$(BUILD_DIR)/sim_main.o: $(SRC_DIR)/main.cpp | $(BUILD_DIR)
	$(HOST_CXX) $(CXXFLAGS) -I$(INC_DIR) -c $< -o $@

$(BUILD_DIR)/sim_npp_processor.o: $(SRC_DIR)/npp_processor.cu | $(BUILD_DIR)
	$(HOST_CXX) $(CXXFLAGS) -x c++ -I$(INC_DIR) -c $< -o $@

$(BUILD_DIR)/sim_custom_kernel.o: $(SRC_DIR)/custom_kernel.cu | $(BUILD_DIR)
	$(HOST_CXX) $(CXXFLAGS) -x c++ -I$(INC_DIR) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

$(ARTIFACTS_DIR):
	mkdir -p $(ARTIFACTS_DIR)

run: all | $(OUTPUT_DIR)
	@echo "Executing CUDA Image Processing Pipeline on $(DATA_DIR)..."
	$(TARGET) --input_dir $(DATA_DIR) --output_dir $(OUTPUT_DIR) --filter all --benchmark

test: sim | $(OUTPUT_DIR)
	@echo "Executing Pipeline in Simulation Verification Mode..."
	$(SIM_TARGET) --input_dir $(DATA_DIR) --output_dir $(OUTPUT_DIR) --filter all --benchmark

benchmark: all | $(OUTPUT_DIR) $(ARTIFACTS_DIR)
	@echo "Executing Benchmark Suite and Saving Execution Log..."
	$(TARGET) --input_dir $(DATA_DIR) --output_dir $(OUTPUT_DIR) --filter all --benchmark | tee $(ARTIFACTS_DIR)/execution_log.txt

package: | $(ARTIFACTS_DIR)
	@echo "Generating Submission Artifact Archives (tar.gz and zip)..."
	tar -czvf $(ARTIFACTS_DIR)/execution_artifacts.tar.gz \
		$(ARTIFACTS_DIR)/execution_log.txt \
		$(ARTIFACTS_DIR)/benchmark_report.md \
		$(OUTPUT_DIR)/*.pgm
	zip -r $(ARTIFACTS_DIR)/execution_artifacts.zip \
		$(ARTIFACTS_DIR)/execution_log.txt \
		$(ARTIFACTS_DIR)/benchmark_report.md \
		$(OUTPUT_DIR)/*.pgm
	@echo "Packaging completed: $(ARTIFACTS_DIR)/execution_artifacts.tar.gz"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR) $(BIN_DIR) $(OUTPUT_DIR)
	rm -f $(ARTIFACTS_DIR)/*.tar.gz $(ARTIFACTS_DIR)/*.zip

help:
	@echo "Available build targets:"
	@echo "  make all       - Build CUDA/NPP executable with nvcc"
	@echo "  make sim       - Build CPU simulation executable with host C++ compiler"
	@echo "  make run       - Execute CUDA pipeline across data/input/"
	@echo "  make test      - Run verification test with CPU simulation binary"
	@echo "  make benchmark - Run pipeline and generate execution_log.txt in artifacts/"
	@echo "  make package   - Create submission archives (.tar.gz and .zip)"
	@echo "  make clean     - Remove all compiled objects, binaries, and generated outputs"
