cmake_minimum_required(VERSION 3.10)
project(NumberRecognition)

find_package(Eigen3 REQUIRED)
add_executable(number_recognition main.cpp util.cpp neural_net.cpp mnist_loader.cpp) 
target_link_libraries(number_recognition Eigen3::Eigen)