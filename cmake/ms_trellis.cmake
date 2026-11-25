# Optional CMake module for Amperite MS (Mad Shift) 1MAD trellis tests.
# Add `include(cmake/ms_trellis.cmake)` to CMakeLists.txt when BUILD_TESTING is enabled.

if(BUILD_TESTING)
  add_executable(test_ms_1mad tests/test_ms_1mad.cpp)
  target_include_directories(test_ms_1mad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_options(test_ms_1mad PRIVATE ${_isa_avx2_flags})
  basalt_embed_manifest(test_ms_1mad)
  basalt_cpu_test(NAME test_ms_1mad COMMAND test_ms_1mad)

  add_executable(test_ms_1mad_cuda tests/test_ms_1mad_cuda.cu)
  target_include_directories(test_ms_1mad_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${_gpu_include_directories})
  target_link_libraries(test_ms_1mad_cuda PRIVATE ${_gpu_runtime_target})
  basalt_embed_manifest(test_ms_1mad_cuda)
  basalt_gpu_test(NAME test_ms_1mad_cuda COMMAND test_ms_1mad_cuda)
endif()
