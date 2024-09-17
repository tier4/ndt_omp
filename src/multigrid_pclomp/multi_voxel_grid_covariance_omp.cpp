#include <autoware/ndt_omp/multigrid_pclomp/multi_voxel_grid_covariance_omp_impl.hpp>

#include <autoware/ndt_omp/multigrid_pclomp/multi_voxel_grid_covariance_omp.h>

template class autoware::ndt_omp::pclomp::MultiVoxelGridCovariance<pcl::PointXYZ>;
template class autoware::ndt_omp::pclomp::MultiVoxelGridCovariance<pcl::PointXYZI>;
