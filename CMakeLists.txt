cmake_minimum_required(VERSION 3.10)
project(NumberRecognition)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenSSL REQUIRED)

# 查找asio库（优先用find_package，如果找不到则手动添加路径）
find_path(ASIO_INCLUDE_DIR asio.hpp
    /usr/include
    /usr/local/include
    /opt/homebrew/include
)
if(NOT ASIO_INCLUDE_DIR)
    message(FATAL_ERROR "asio.hpp not found. 请先安装asio库，例如: brew install asio")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})  # 关键：添加 Eigen3 头文件路径
include_directories(${ASIO_INCLUDE_DIR})    # 关键：添加asio头文件路径

add_executable(number_recognition main.cpp mnist_loader.cpp neural_net.cpp util.cpp web_server.cpp)

target_link_libraries(number_recognition
    ${OpenCV_LIBS}
    OpenSSL::SSL
    OpenSSL::Crypto
    pthread
    Eigen3::Eigen  # 关键：链接 Eigen3
)
include_directories(./)