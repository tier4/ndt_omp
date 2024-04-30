cmake_minimum_required(VERSION 3.5)
project(ndt_omp)

add_definitions(-std=c++17)
set(CMAKE_CXX_FLAGS "-std=c++17")

# Compile flags for SIMD instructions
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
  # For x86_64 architecture, SIMD instruction set is fixed below versions,
  add_compile_options(-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2)
else()
  # For other architecture, like arm64, compile flags are generally prepared by compiler
  # march=native is disabled as default for specific depending pcl libraries
  # or pre-building packages for other computers.
  if(BUILD_WITH_MARCH_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()

# pcl 1.7 causes a segfault when it is built with debug mode
set(CMAKE_BUILD_TYPE "RELEASE")

# ROS2
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

message(STATUS "PCL_INCLUDE_DIRS:" ${PCL_INCLUDE_DIRS})
message(STATUS "PCL_LIBRARY_DIRS:" ${PCL_LIBRARY_DIRS})
message(STATUS "PCL_DEFINITIONS:" ${PCL_DEFINITIONS})

find_package(OpenMP)

###########
## Build ##
###########

ament_auto_add_library(ndt_omp SHARED
  src/pclomp/voxel_grid_covariance_omp.cpp
  src/pclomp/ndt_omp.cpp
  src/pclomp/gicp_omp.cpp
)

target_link_libraries(ndt_omp ${PCL_LIBRARIES})

if(OpenMP_CXX_FOUND)
  target_link_libraries(ndt_omp OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP not found")
endif()

#############################
## ROS 2 multigrid ndt_omp ##
#############################
ament_auto_add_library(multigrid_ndt_omp SHARED
  src/multigrid_pclomp/multi_voxel_grid_covariance_omp.cpp
  src/multigrid_pclomp/multigrid_ndt_omp.cpp
  src/estimate_covariance/estimate_covariance.cpp
)
target_link_libraries(multigrid_ndt_omp ${PCL_LIBRARIES})
if(OpenMP_CXX_FOUND)
  target_link_libraries(multigrid_ndt_omp OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP not found")
endif()
###################################
## ROS 2 multigrid ndt_omp (end) ##
###################################

include_directories(
  ${PROJECT_SOURCE_DIR}/include
)

add_executable(align
  apps/align.cpp
)
add_dependencies(align
  ndt_omp multigrid_ndt_omp
)
target_link_libraries(align
  ${PCL_LIBRARIES}
  ndt_omp multigrid_ndt_omp
)

# regression test
add_executable(regression_test
  apps/regression_test.cpp
)
add_dependencies(regression_test
  ndt_omp multigrid_ndt_omp
)
target_link_libraries(regression_test
  ${PCL_LIBRARIES}
  ndt_omp multigrid_ndt_omp
)

# check_covariance
add_executable(check_covariance
  apps/check_covariance.cpp
)
add_dependencies(check_covariance
  ndt_omp multigrid_ndt_omp
)
target_link_libraries(check_covariance
  ${PCL_LIBRARIES}
  ndt_omp multigrid_ndt_omp
)

# initialpose estimation
ament_auto_add_executable(initialpose_estimation
  apps/initialpose_estimation.cpp
  src/bbs3d/bbs3d.cpp
  src/bbs3d/voxelmaps.cpp
  src/bbs3d/bbs3d_search.cpp
  src/bbs3d/random_search.cpp
)
add_dependencies(initialpose_estimation
  ndt_omp multigrid_ndt_omp
)
target_link_libraries(initialpose_estimation
  ${PCL_LIBRARIES}
  ndt_omp multigrid_ndt_omp
)

ament_auto_package()
