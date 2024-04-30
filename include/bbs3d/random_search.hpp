#ifndef RANDOM_SEARCH_HPP
#define RANDOM_SEARCH_HPP

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <pcl/point_types.h>

#include "multigrid_pclomp/multigrid_ndt_omp.h"

namespace random_search {
  using PointSource = pcl::PointXYZ;
  using PointTarget = pcl::PointXYZ;
  using NormalDistributionsTransform = pclomp::MultiGridNormalDistributionsTransform<PointSource, PointTarget>;

  geometry_msgs::msg::PoseWithCovarianceStamped random_search(std::shared_ptr<NormalDistributionsTransform> ndt_ptr, const geometry_msgs::msg::PoseWithCovarianceStamped& initial_pose_with_cov, const int64_t particles_num);

}  // namespace random_search

#endif  // RANDOM_SEARCH_HPP
