cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(pytorch_project)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(pytorch_project main.cpp)

target_link_libraries(pytorch_project "${TORCH_LIBRARIES}")
set_property(TARGET pytorch_project PROPERTY CXX_STANDARD 17)