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
# cuest_wrapper — minimal C++ RAII interface over the cuEST C API
# (no JSON / XYZ I/O / SCF — those live in the application)
# ---------------------------------------------------------------------------
add_library(cuest_wrapper STATIC
  cuest_wrapper/src/integrals.cpp
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
  CUDA::nvtx3
  ${CUEST_LIBRARY}
  m
)

target_compile_options(cuest_wrapper PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(cuest_wrapper PRIVATE
  $<$<CONFIG:RELEASE>:-O3 -DNDEBUG -march=native>
)

# ---------------------------------------------------------------------------
# cuest_dft — CLI driver (JSON/XYZ I/O, SCF, basis-from-BSE)
# ---------------------------------------------------------------------------
add_executable(cuest_dft
  src/main.cpp
  src/scf.cpp
  src/diis.cpp
  src/sad_guess.cpp
  src/atomic_config.cpp
  src/atomic_symmetry.cpp
  src/basis_from_json.cpp
  src/basis_ecp_from_json.cpp
)

target_include_directories(cuest_dft PRIVATE
  ${CMAKE_SOURCE_DIR}/src
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

# ---------------------------------------------------------------------------
# probe_symmetry — validates the Cartesian rotation representation against a
# real cuEST overlap matrix (see tools/probe_symmetry.cpp).
# ---------------------------------------------------------------------------
add_executable(probe_symmetry
  tools/probe_symmetry.cpp
  src/atomic_symmetry.cpp
  src/basis_from_json.cpp
  src/basis_ecp_from_json.cpp
)
target_include_directories(probe_symmetry PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(probe_symmetry PRIVATE cuest_wrapper)
target_compile_options(probe_symmetry PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(probe_symmetry PROPERTIES
  BUILD_RPATH "${CUEST_LIB_DIR}"
  INSTALL_RPATH "${CUEST_LIB_DIR}")

# ---------------------------------------------------------------------------
# probe_atom — runs one element's spherically symmetric free-atom reference,
# for comparison against PySCF's AtomSphAverageRHF (see test/validate_sad.py).
# ---------------------------------------------------------------------------
add_executable(probe_atom
  tools/probe_atom.cpp
  src/sad_guess.cpp
  src/atomic_symmetry.cpp
  src/atomic_config.cpp
  src/basis_from_json.cpp
  src/basis_ecp_from_json.cpp
)
target_include_directories(probe_atom PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(probe_atom PRIVATE cuest_wrapper)
target_compile_options(probe_atom PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(probe_atom PROPERTIES
  BUILD_RPATH "${CUEST_LIB_DIR}"
  INSTALL_RPATH "${CUEST_LIB_DIR}")

# ---------------------------------------------------------------------------
# test_atomic_symmetry — GPU-free unit test for the rotation representation and
# the commutant projector (tools/test_atomic_symmetry.cpp).
# ---------------------------------------------------------------------------
add_executable(test_atomic_symmetry
  tools/test_atomic_symmetry.cpp
  src/atomic_symmetry.cpp
)
target_include_directories(test_atomic_symmetry PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(test_atomic_symmetry PRIVATE -Wall -Wextra -Wpedantic)

# Install
install(TARGETS cuest_dft DESTINATION bin)
install(TARGETS cuest_wrapper ARCHIVE DESTINATION lib)
install(DIRECTORY cuest_wrapper/include/ DESTINATION include)
