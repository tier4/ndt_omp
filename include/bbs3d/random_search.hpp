// Copyright 2024 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
