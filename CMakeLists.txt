cmake_minimum_required(VERSION 3.8)
project(AirIO_Damvi)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(ONNXRUNTIME_ROOT /opt/onnxruntime)
set(CUDA_INCLUDE_DIR "/usr/local/cuda/include")

# default ON
option(ENABLE_TENSORRT "Build TensorRT node (requires CUDA+TensorRT)" ON)

# find dependencies (common)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

add_library(imu_buffer_lib include/core/imu_buffer.cpp)
target_include_directories(imu_buffer_lib PUBLIC include)

# Velocity EKF library 
add_library(velocity_ekf_lib include/airio/ekf/velocity_ekf.cpp)
target_include_directories(velocity_ekf_lib PUBLIC include)
target_link_libraries(velocity_ekf_lib PUBLIC Eigen3::Eigen)


# -------------------------
# ONNX Runtime node
# -------------------------
add_executable(onnx_airio_Damvi src/onnx_airio_Damvi.cpp)
ament_target_dependencies(onnx_airio_Damvi rclcpp sensor_msgs)
target_link_libraries(onnx_airio_Damvi 
  imu_buffer_lib 
  onnxruntime 
  velocity_ekf_lib
)
set_target_properties(onnx_airio_Damvi PROPERTIES
  BUILD_RPATH   "${ONNXRUNTIME_ROOT}/lib"
  INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
)

install(TARGETS onnx_airio_Damvi imu_buffer_lib
  DESTINATION lib/${PROJECT_NAME}
)

# -------------------------
# TensorRT node
# -------------------------
if(ENABLE_TENSORRT)

  find_package(CUDAToolkit REQUIRED)

  find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS
      /usr/include
      /usr/include/x86_64-linux-gnu
      /usr/local/include
  )

  find_library(TENSORRT_NVINFER_LIB
    NAMES nvinfer
    HINTS
      /usr/lib
      /usr/lib/x86_64-linux-gnu
      /usr/local/lib
      /usr/local/lib64
  )

  find_library(TENSORRT_NVONNXPARSER_LIB
    NAMES nvonnxparser
    HINTS
      /usr/lib
      /usr/lib/x86_64-linux-gnu
      /usr/local/lib
      /usr/local/lib64
  )

  find_library(TENSORRT_NVINFER_PLUGIN_LIB
    NAMES nvinfer_plugin
    HINTS
      /usr/lib
      /usr/lib/x86_64-linux-gnu
      /usr/local/lib
      /usr/local/lib64
  )

  # CUDA runtime (for cudaMalloc/cudaMemcpy etc.)
  find_library(CUDA_CUDART_LIB
    NAMES cudart
    HINTS
      /usr/local/cuda/lib64
      /usr/local/cuda/lib
      /usr/lib
      /usr/lib/x86_64-linux-gnu
  )

  # 빌드 타임 sanity check (TRT 켰을 때만 강제)
  if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIB OR NOT TENSORRT_NVONNXPARSER_LIB)
    message(FATAL_ERROR
      "TensorRT not found but ENABLE_TENSORRT=ON.\n"
      "  TENSORRT_INCLUDE_DIR=${TENSORRT_INCLUDE_DIR}\n"
      "  TENSORRT_NVINFER_LIB=${TENSORRT_NVINFER_LIB}\n"
      "  TENSORRT_NVONNXPARSER_LIB=${TENSORRT_NVONNXPARSER_LIB}\n"
    )
  endif()

  add_executable(tensorrt_airio_Damvi src/tensorrt_airio_Damvi.cpp)
  ament_target_dependencies(tensorrt_airio_Damvi rclcpp sensor_msgs)
  target_include_directories(tensorrt_airio_Damvi PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
  )
  target_link_libraries(tensorrt_airio_Damvi
    imu_buffer_lib
    velocity_ekf_lib
    ${TENSORRT_NVINFER_LIB}
    ${TENSORRT_NVONNXPARSER_LIB}
    ${TENSORRT_NVINFER_PLUGIN_LIB}
    ${CUDA_CUDART_LIB}
  )

  get_filename_component(_TRT_LIB_DIR "${TENSORRT_NVINFER_LIB}" DIRECTORY)
  set_target_properties(tensorrt_airio_Damvi PROPERTIES
    BUILD_RPATH   "${_TRT_LIB_DIR}"
    INSTALL_RPATH "${_TRT_LIB_DIR}"
  )

  install(TARGETS tensorrt_airio_Damvi
    DESTINATION lib/${PROJECT_NAME}
  )

else()
  message(STATUS "ENABLE_TENSORRT=OFF -> Skip building tensorrt_airio_Damvi")
endif()

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
