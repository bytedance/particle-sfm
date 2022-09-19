cmake_minimum_required(VERSION 3.4)
project(traj_ceres)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(SuiteSparse)
find_package(CXSparse)
include_directories(${CXSPARSE_INCLUDE_DIRS})

add_subdirectory(third_party)
add_subdirectory(src)

include_directories(${HDF5_INCLUDE_DIRS})

pybind11_add_module(particlesfm ${PARTICLESFM_SOURCES})
target_link_libraries(particlesfm PRIVATE HighFive ${HDF5_C_LIBRARIES} ${CERES_LIBRARIES} ${GLOG_LIBRARIES} ${SUITESPARSE_LIBRARIES} ${CXSPARSE_LIBRARIES})

