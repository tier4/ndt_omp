#include <autoware/ndt_omp/pclomp/ndt_omp_impl.hpp>

#include <autoware/ndt_omp/pclomp/ndt_omp.h>

template class autoware::ndt_omp::pclomp::NormalDistributionsTransform<
  pcl::PointXYZ, pcl::PointXYZ>;
template class autoware::ndt_omp::pclomp::NormalDistributionsTransform<
  pcl::PointXYZI, pcl::PointXYZI>;
