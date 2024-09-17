// this cannot be swapped
// clang-format off
#include <autoware/ndt_omp/pclomp/gicp_omp.h>
#include <autoware/ndt_omp/pclomp/gicp_omp_impl.hpp>
// clang-format on

template class autoware::ndt_omp::pclomp::GeneralizedIterativeClosestPoint<
  pcl::PointXYZ, pcl::PointXYZ>;
template class autoware::ndt_omp::pclomp::GeneralizedIterativeClosestPoint<
  pcl::PointXYZI, pcl::PointXYZI>;
