cmake_minimum_required(VERSION 3.31.5)

project(multiply LANGUAGES CXX CUDA)

file(GLOB_RECURSE SOURCE_FILES src/*.cpp src/*.cu src/*.h src/*.cuh)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(multiply ${SOURCE_FILES})