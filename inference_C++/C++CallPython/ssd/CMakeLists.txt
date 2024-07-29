cmake_minimum_required(VERSION 2.8)

project(project_ssd)


set(SDK_VERSION 0_0_1)
######################## build type ###############################
set(CMAKE_BUILD_TYPE "Release")				# 指定生成的版本
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
###################################################################


############################## CXX11 ##############################
set(CMAKE_CXX_STANDARD 11)					# C++ 11 编译器
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
###################################################################


############################# Python3 #############################
set(PYTHON_ROOT "/usr")
message(STATUS ${PYTHON_ROOT})
include_directories(${PYTHON_ROOT}/include/python3.6/)
link_directories(${PYTHON_ROOT}/lib/python3.6/config-3.6m-x86_64-linux-gnu/)
###################################################################


############################ OpenCV4.1 ############################
set(OpenCV_DIR "/usr/local")
message(STATUS ${OpenCV_DIR})
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_DIR}/include/opencv4/opencv2)
    include_directories( ${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_DIR}/lib)
    message(STATUS "Configure package with OpenCV!")
    set(HAVE_OpenCV True)
else()
    set(HAVE_OpenCV False)
    message(STATUS "Configure package without OpenCV!")
endif()

############################ HC Vision ############################


###################################################################
aux_source_directory(./src DIR_SRCS)   
add_executable(TargetDetection  ${DIR_SRCS})
target_link_libraries(TargetDetection -lpython3.6m ${OpenCV_LIBS})


