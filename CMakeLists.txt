cmake_minimum_required(VERSION 3.8)
project(Air-IO_Damvi)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(ONNXRUNTIME_ROOT /opt/onnxruntime)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

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
target_link_libraries(tensorrt_airio_Damvi imu_buffer_lib)

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
