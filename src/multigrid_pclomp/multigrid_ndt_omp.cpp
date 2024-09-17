#include <autoware/ndt_omp/multigrid_pclomp/multigrid_ndt_omp_impl.hpp>

#include <autoware/ndt_omp/multigrid_pclomp/multigrid_ndt_omp.h>

template class autoware::ndt_omp::pclomp::MultiGridNormalDistributionsTransform<
  pcl::PointXYZ, pcl::PointXYZ>;
template class autoware::ndt_omp::pclomp::MultiGridNormalDistributionsTransform<
  pcl::PointXYZI, pcl::PointXYZI>;
