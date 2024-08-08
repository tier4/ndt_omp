#include <multigrid_pclomp/multigrid_ndt_omp_impl.hpp>

#include <multigrid_pclomp/multigrid_ndt_omp.h>

template class pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>;
template class pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>;
