cmake_minimum_required(VERSION 3.8)
project(Air-IO_Damvi)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(ONNXRUNTIME_ROOT /opt/onnxruntime)
set(CUDA_INCLUDE_DIR "/usr/local/cuda/include")
set(TENSORRT_ROOT "" CACHE PATH "TensorRT root path (optional)")

# Try common include/library locations (+ optional TENSORRT_ROOT)
find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS
    ${TENSORRT_ROOT}
    ${TENSORRT_ROOT}/include
    /usr/include
    /usr/include/x86_64-linux-gnu
    /usr/local/include
)

find_library(TENSORRT_NVINFER_LIB
  NAMES nvinfer
  HINTS
    ${TENSORRT_ROOT}
    ${TENSORRT_ROOT}/lib
    ${TENSORRT_ROOT}/lib64
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
    /usr/local/lib64
)

find_library(TENSORRT_NVONNXPARSER_LIB
  NAMES nvonnxparser
  HINTS
    ${TENSORRT_ROOT}
    ${TENSORRT_ROOT}/lib
    ${TENSORRT_ROOT}/lib64
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
    /usr/local/lib64
)

find_library(TENSORRT_NVINFER_PLUGIN_LIB
  NAMES nvinfer_plugin
  HINTS
    ${TENSORRT_ROOT}
    ${TENSORRT_ROOT}/lib
    ${TENSORRT_ROOT}/lib64
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/lib
    /usr/local/lib64
)

# CUDA runtime (for cudaMalloc/cudaMemcpy etc. inside TRT samples/usages)
find_library(CUDA_CUDART_LIB
  NAMES cudart
  HINTS
    /usr/local/cuda/lib64
    /usr/local/cuda/lib
    /usr/lib
    /usr/lib/x86_64-linux-gnu
)

# Build-time sanity check (fails early if TRT not found)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIB OR NOT TENSORRT_NVONNXPARSER_LIB)
  message(FATAL_ERROR
    "TensorRT not found.\n"
    "  TENSORRT_INCLUDE_DIR=${TENSORRT_INCLUDE_DIR}\n"
    "  TENSORRT_NVINFER_LIB=${TENSORRT_NVINFER_LIB}\n"
    "  TENSORRT_NVONNXPARSER_LIB=${TENSORRT_NVONNXPARSER_LIB}\n"
    "Hint: set -DTENSORRT_ROOT=/path/to/tensorrt"
  )
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(
  ${ONNXRUNTIME_ROOT}/include
)
set(ONNXRUNTIME_LIB
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
)
link_directories(
  ${ONNXRUNTIME_ROOT}/lib
)

add_library(imu_buffer_lib include/core/imu_buffer.cpp)
target_include_directories(imu_buffer_lib PUBLIC include)

# ONNX Runtime 기반 노드
add_executable(onnx_airio_Damvi src/onnx_airio_Damvi.cpp)
ament_target_dependencies(onnx_airio_Damvi rclcpp sensor_msgs)
target_link_libraries(onnx_airio_Damvi imu_buffer_lib
  onnxruntime
)
set_target_properties(onnx_airio_Damvi PROPERTIES
  BUILD_RPATH "/opt/onnxruntime/lib"
  INSTALL_RPATH "/opt/onnxruntime/lib"
)
install(TARGETS onnx_airio_Damvi imu_buffer_lib
  DESTINATION lib/${PROJECT_NAME}
)

# TensorRT 기반 노드
add_executable(tensorrt_airio_Damvi src/tensorrt_airio_Damvi.cpp)
ament_target_dependencies(tensorrt_airio_Damvi rclcpp sensor_msgs)
target_include_directories(tensorrt_airio_Damvi PRIVATE
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIR}
)
target_link_libraries(tensorrt_airio_Damvi imu_buffer_lib
  ${TENSORRT_NVINFER_LIB}
  ${TENSORRT_NVONNXPARSER_LIB}
  ${TENSORRT_NVINFER_PLUGIN_LIB}
  ${CUDA_CUDART_LIB}
)
get_filename_component(_TRT_LIB_DIR "${TENSORRT_NVINFER_LIB}" DIRECTORY)
set_target_properties(tensorrt_airio_Damvi PROPERTIES
  BUILD_RPATH "${_TRT_LIB_DIR}"
  INSTALL_RPATH "${_TRT_LIB_DIR}"
)
install(TARGETS tensorrt_airio_Damvi
  DESTINATION lib/${PROJECT_NAME}
)

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
