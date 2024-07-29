cmake_minimum_required(VERSION 3.9)

project(add)

set(SDK_VERSION 0_0_1)
# >>> build type 
set(CMAKE_BUILD_TYPE "Release")				# 指定生成的版本
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
# <<<


# >>> CXX11 
set(CMAKE_CXX_STANDARD 11)				# C++ 11 编译器
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# <<< 


# >>> Python3 
set(PYTHON_ROOT "/home/heygears/anaconda3/envs/MeshCNN")
message("python root: " ${PYTHON_ROOT})
include_directories(${PYTHON_ROOT}/include/)
link_directories(${PYTHON_ROOT}/lib/)
# <<<

# --- generate ---
add_executable(add add.cpp)
target_link_libraries(add -lpython3.6m)

