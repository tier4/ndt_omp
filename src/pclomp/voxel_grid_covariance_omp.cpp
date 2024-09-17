#include <autoware/ndt_omp/pclomp/voxel_grid_covariance_omp_impl.hpp>

#include <autoware/ndt_omp/pclomp/voxel_grid_covariance_omp.h>

template class autoware::ndt_omp::pclomp::VoxelGridCovariance<pcl::PointXYZ>;
template class autoware::ndt_omp::pclomp::VoxelGridCovariance<pcl::PointXYZI>;
