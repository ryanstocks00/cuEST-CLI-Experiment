cmake_minimum_required(VERSION 3.20)
project(cuest_dft LANGUAGES CXX CUDA VERSION 0.1.0
  DESCRIPTION "DFT package using NVIDIA cuEST with ECP support")

# C++20 / CUDA 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Default GPU architectures:
#   80 = A100, 89 = Ada (RTX 40), 90 = H100
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 89 90)
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

# ---------------------------------------------------------------------------
# cuest_wrapper — standalone RAII library around the cuEST C API
# ---------------------------------------------------------------------------
add_library(cuest_wrapper STATIC
  cuest_wrapper/src/basis_nvidia.cpp
  cuest_wrapper/src/basis_ecp.cpp
  cuest_wrapper/src/integrals.cpp
  cuest_wrapper/src/scf.cpp
)

target_include_directories(cuest_wrapper PUBLIC
  ${CMAKE_SOURCE_DIR}/cuest_wrapper/include
  ${CMAKE_SOURCE_DIR}/external
  ${CUEST_INCLUDE_DIR}
)

target_compile_definitions(cuest_wrapper PUBLIC
  CUESTAPI
  $<$<BOOL:${CUEST_ENABLE_ECP}>:CUEST_ENABLE_ECP>
  $<$<BOOL:${CUEST_ENABLE_JIT}>:CUEST_ENABLE_JIT>
)

target_link_libraries(cuest_wrapper PUBLIC
  CUDA::cublas
  CUDA::cusolver
  CUDA::cudart
  CUDA::cuda_driver
  ${CUEST_LIBRARY}
  m
)

target_compile_options(cuest_wrapper PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(cuest_wrapper PRIVATE
  $<$<CONFIG:RELEASE>:-O3 -DNDEBUG -march=native>
)

# ---------------------------------------------------------------------------
# cuest_dft — CLI driver
# ---------------------------------------------------------------------------
add_executable(cuest_dft
  src/main.cpp
)

target_link_libraries(cuest_dft PRIVATE cuest_wrapper)

target_compile_options(cuest_dft PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(cuest_dft PRIVATE
  $<$<CONFIG:RELEASE>:-O3 -DNDEBUG -march=native>
)

# Set runtime library path for libcuest.so
set_target_properties(cuest_dft PROPERTIES
  BUILD_RPATH "${CUEST_LIB_DIR}"
  INSTALL_RPATH "${CUEST_LIB_DIR}"
)

# Install
install(TARGETS cuest_dft DESTINATION bin)
install(TARGETS cuest_wrapper ARCHIVE DESTINATION lib)
install(DIRECTORY cuest_wrapper/include/ DESTINATION include)
