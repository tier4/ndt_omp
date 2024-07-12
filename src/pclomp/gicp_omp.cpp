// this cannot be swapped
// clang-format off
#include <pclomp/gicp_omp.h>
#include <pclomp/gicp_omp_impl.hpp>
// clang-format on

template class pclomp::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>;
template class pclomp::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI>;
