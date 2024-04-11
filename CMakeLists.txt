cmake_minimum_required(VERSION 3.0)

project(MyONNXRuntimeOpenCVProject)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 寻找 OpenCV 库
find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "Found OpenCV")
    message(STATUS "Opencv Version:${OpenCV_VERSION}")
    message(STATUS "OpenCV Library: ${OpenCV_LIBS}")
    message(STATUS "Include path: ${OpenCV_INCLUDE_PATH}")
else()
    message(FATAL_ERROT "Could not find OpenCV library")
endif()

# 如果你已经安装了 ONNX Runtime 并且它在标准路径下，你可以这样找到它
# 否则你需要指定它的路径
# find_package(ONNXRuntime REQUIRED)
set(ONNXRUNTIME_ROOT_PATH /下载解压的路径/onnxruntime-linux-x64-gpu-1.16.3)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so)
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})

add_executable(main inference_cpu.cpp)
# add_executable(main inference_cuda.cpp)
#add_executable(main inference_fuc.cpp)


target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})


target_link_libraries(main ${OpenCV_LIBS} ${ONNXRUNTIME_LIB} )
