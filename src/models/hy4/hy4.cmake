# Hy4-flash (HYV4FlashForCausalLM) targets: the decode kernels and their parity test.

add_library(model_hy4_kernels STATIC
  src/models/hy4/kernels/decode.cu
)
target_include_directories(model_hy4_kernels PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
set_target_properties(model_hy4_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

add_library(model_hy4 STATIC
  src/models/hy4/engine.cpp
  src/models/hy4/adapter.cpp
)
target_include_directories(model_hy4 PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(model_hy4 PUBLIC model_hy4_kernels core kernels ggml_quants platform model CUDA::cudart)

add_executable(hy4_generate src/models/hy4/hy4_generate.cpp)
target_link_libraries(hy4_generate PRIVATE model_hy4 spec)

add_executable(hy4_kernels_parity src/models/hy4/kernels/parity.cpp)
target_link_libraries(hy4_kernels_parity PRIVATE model_hy4_kernels CUDA::cudart)
basalt_gpu_test(NAME hy4_kernels_parity COMMAND hy4_kernels_parity)
set_tests_properties(hy4_kernels_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
