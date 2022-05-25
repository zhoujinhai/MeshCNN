cmake_minimum_required(VERSION 3.9)

project(mesh VERSION 1.0)

# >>> build type 
set(CMAKE_BUILD_TYPE "Release")				# 指定生成的版本
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
# <<<


# >>> CXX11 
set(CMAKE_CXX_STANDARD 11)				# C++ 11 编译器
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# <<< CXX11


# >>> Python3 
set(PYTHON_ROOT "/home/zjh/anaconda3/envs/MeshCNN")
message("python root: " ${PYTHON_ROOT})
include_directories(${PYTHON_ROOT}/include/)
include_directories(${PYTHON_ROOT}/include/python3.6m/)
link_directories(${PYTHON_ROOT}/lib/)
# <<< Python3

# >>> Copy file
file(COPY ./mesh_net DESTINATION ./)
file(COPY ./class_features_info DESTINATION ./)
file(COPY ./model_weight DESTINATION ./)
# <<< Copy file

# --- generate ---
add_executable(meshCNN meshCNN.cpp)
target_link_libraries(meshCNN -lpython3.6m)

