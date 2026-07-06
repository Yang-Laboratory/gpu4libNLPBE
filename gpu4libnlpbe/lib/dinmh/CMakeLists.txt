set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
set(CMAKE_VERBOSE_MAKEFILE ON)
add_library(dinmh SHARED amgcl_wrapper.cu)

# Fetch the header-only amgcl library into the build tree so the caller does
# not need to pass -DAMGCL_INCLUDE_DIR. The clone lands at
# ${CMAKE_BINARY_DIR}/_deps/amgcl-src and is reused on subsequent builds.
include(FetchContent)
FetchContent_Declare(amgcl
  GIT_REPOSITORY https://github.com/ddemidov/amgcl.git
  GIT_TAG        master
  GIT_SHALLOW    TRUE)
FetchContent_GetProperties(amgcl)
if(NOT amgcl_POPULATED)
  FetchContent_Populate(amgcl)
endif()
set(AMGCL_INCLUDE_DIR ${amgcl_SOURCE_DIR})

set_target_properties(dinmh PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR} CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(dinmh PUBLIC ${AMGCL_INCLUDE_DIR})
target_link_libraries(dinmh ${OPENMP_CXX_PROPERTIES} CUDA::cusparse)
target_compile_options(dinmh PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>)
