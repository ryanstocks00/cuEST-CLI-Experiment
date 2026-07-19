cmake_minimum_required(VERSION 3.20)
project(cuest_dft LANGUAGES C CXX CUDA VERSION 0.1.0
  DESCRIPTION "DFT package using NVIDIA cuEST with ECP support")

# C++17 required for structured bindings, string_view, etc.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Default GPU architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 90)
endif()

# cuEST paths
set(CUEST_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/cuest_include"
  CACHE PATH "Path to cuEST headers")
set(CUEST_LIB_DIR "${CMAKE_SOURCE_DIR}/external" CACHE PATH "Path to cuEST libraries")

# Find cuEST library
find_library(CUEST_LIBRARY cuest
  HINTS ${CUEST_LIB_DIR}
  REQUIRED)

# cuEST options
option(CUEST_ENABLE_ECP "Enable ECP support" ON)
option(CUEST_ENABLE_JIT "Enable JIT compilation (requires CUDA 13+)" ON)
option(CUEST_USE_STATIC "Use static cuEST library" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Main executable
add_executable(cuest_dft
  src/main.cpp
  src/basis_nvidia.cpp
  src/basis_ecp.cpp
  src/integrals.cpp
  src/scf.cpp
  src/nvidia_df_wrapper.c
)

set_source_files_properties(src/nvidia_df_wrapper.c PROPERTIES
  COMPILE_OPTIONS "-std=gnu11;-DCUESTAPI"
  INCLUDE_DIRECTORIES "${CUEST_INCLUDE_DIR};${CMAKE_SOURCE_DIR}/temp_samples/cuEST/c_examples/common"
)
target_include_directories(cuest_dft PRIVATE
  ${CMAKE_SOURCE_DIR}/temp_samples/cuEST/c_examples/common
)

# cuEST wrapper library (header-only mostly, but compiled helpers)
add_library(cuest_wrapper INTERFACE)
target_include_directories(cuest_wrapper INTERFACE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/temp_samples/cuEST/c_examples/common
  ${CUEST_INCLUDE_DIR}
)
target_compile_definitions(cuest_wrapper INTERFACE
  CUESTAPI
  $<$<BOOL:${CUEST_ENABLE_ECP}>:CUEST_ENABLE_ECP>
  $<$<BOOL:${CUEST_ENABLE_JIT}>:CUEST_ENABLE_JIT>
)

target_link_libraries(cuest_wrapper INTERFACE
  CUDA::cublas
  CUDA::cusolver
  CUDA::cudart
  CUDA::cuda_driver
  ${CUEST_LIBRARY}
)

target_link_libraries(cuest_dft PRIVATE cuest_wrapper)

# Link math library
target_link_libraries(cuest_dft PRIVATE m)

# Set runtime library path
set_target_properties(cuest_dft PROPERTIES
  BUILD_RPATH "${CUEST_LIB_DIR}"
  INSTALL_RPATH "${CUEST_LIB_DIR}"
)

# Install
install(TARGETS cuest_dft DESTINATION bin)
