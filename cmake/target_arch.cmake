# cmake/target_arch.cmake - the two things a build is for: a CPU architecture and one GPU compute capability.
# Included right after project() by the root project and by the standalone tools/vision configuration.
#
#   BASALT_ARM64    ON when the target is Arm64: the CPU kernels are NEON. OFF on x86-64 (AVX2 / AVX-512).
#   BASALT_CUDA_CC  86 (Ampere), the one compute capability the kernels are built for (include/basalt/cuda_target.hpp).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$" OR "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" MATCHES "^ARM64")
  set(BASALT_ARM64 ON)
else()
  set(BASALT_ARM64 OFF)
endif()

if(_basalt_cuda_arch_default)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
endif()
# Each entry must be the supported architecture; the kernels use its instructions and nothing else is tested.
foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
  string(REGEX REPLACE "-(real|virtual)$" "" _base "${_a}")
  if(NOT _base MATCHES "^86$")
    message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}': Amperite builds for compute capability 8.6 only")
  endif()
endforeach()
set(BASALT_CUDA_CC 86)
add_compile_definitions(BASALT_CUDA_CC=${BASALT_CUDA_CC})
if(BASALT_ARM64)
  include(${CMAKE_CURRENT_LIST_DIR}/arm64.cmake)
  basalt_arm64_global_flags()
  message(STATUS "basalt: Arm64, NEON CPU kernels; CUDA arch ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "basalt: x86-64, AVX2 / AVX-512 CPU kernels; CUDA arch ${CMAKE_CUDA_ARCHITECTURES}")
endif()
