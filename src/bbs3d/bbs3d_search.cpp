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

#include "bbs3d/initialpose_estimation.hpp"
#include "bbs3d/bbs3d.hpp"

#include <pcl/filters/passthrough.h>

namespace initialpose_estimation {

SearchResult bbs3d_search(std::shared_ptr<NormalDistributionsTransform> ndt_ptr, const geometry_msgs::msg::PoseWithCovarianceStamped& initial_pose_with_cov) {
  const Eigen::Matrix4f initial_pose = pose_to_matrix4f(initial_pose_with_cov.pose.pose);
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr source_cloud = ndt_ptr->getInputSource();
  std::cout << "source_cloud->size(): " << source_cloud->size() << std::endl;

  BBS3D bbs3d;

  std::vector<Eigen::Vector3d> src_points;
  double max_norm = 0.0;
  for(const auto& point : source_cloud->points) {
    src_points.emplace_back(point.x, point.y, point.z);
    max_norm = std::max(max_norm, src_points.back().norm());
  }
  bbs3d.set_src_points(src_points);

  const double kSearchWidth = 10.0;
  const double cloud_width = kSearchWidth + max_norm;

  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr target_cloud_const = ndt_ptr->getInputTarget();
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>(*target_cloud_const));

  std::cout << "target_cloud->size(): " << target_cloud->size() << std::endl;
  // filter target_cloud
  pcl::PassThrough<pcl::PointXYZ> pass_x;
  pass_x.setInputCloud(target_cloud);
  pass_x.setFilterFieldName("x");
  pass_x.setFilterLimits(initial_pose(0, 3) - cloud_width, initial_pose(0, 3) + cloud_width);
  pass_x.filter(*target_cloud);
  pcl::PassThrough<pcl::PointXYZ> pass_y;
  pass_y.setInputCloud(target_cloud);
  pass_y.setFilterFieldName("y");
  pass_y.setFilterLimits(initial_pose(1, 3) - cloud_width, initial_pose(1, 3) + cloud_width);
  pass_y.filter(*target_cloud);
  std::cout << "target_cloud->size(): " << target_cloud->size() << std::endl;

  std::vector<Eigen::Vector3d> target_points;
  for(const auto& point : target_cloud->points) {
    target_points.emplace_back(point.x, point.y, point.z);
  }

  const double min_level_res = 2.0;
  const int max_level = 2;
  bbs3d.set_tar_points(target_points, min_level_res, max_level);

  // other settings
  bbs3d.set_score_threshold_percentage(0.1);
  bbs3d.enable_timeout();

  // set search range
  Eigen::Vector3d min_xyz = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
  Eigen::Vector3d max_xyz = Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());
  for(const auto& point : target_points) {
    min_xyz = min_xyz.cwiseMin(point);
    max_xyz = max_xyz.cwiseMax(point);
  }
  std::cout << "min_xyz: " << min_xyz.transpose() << std::endl;
  std::cout << "max_xyz: " << max_xyz.transpose() << std::endl;
  std::cout << "gt_pose: " << initial_pose(0, 3) << " " << initial_pose(1, 3) << " " << initial_pose(2, 3) << std::endl;
  min_xyz.x() = initial_pose(0, 3) - kSearchWidth;
  min_xyz.y() = initial_pose(1, 3) - kSearchWidth;
  max_xyz.x() = initial_pose(0, 3) + kSearchWidth;
  max_xyz.y() = initial_pose(1, 3) + kSearchWidth;
  bbs3d.set_trans_search_range(min_xyz, max_xyz);

  bbs3d.localize();

  const int best_score = bbs3d.get_best_score();
  std::cout << "best_score: " << best_score << std::endl;
  std::cout << "best_score_percentage: " << bbs3d.get_best_score_percentage() << std::endl;

  const Eigen::Matrix4d global_pose = bbs3d.get_global_pose();

  std::cout << std::fixed;
  std::cout << "gt_pose   = " << initial_pose(0, 3) << ", " << initial_pose(1, 3) << ", " << initial_pose(2, 3) << std::endl;
  std::cout << "pred_pose = " << global_pose(0, 3) << ", " << global_pose(1, 3) << ", " << global_pose(2, 3) << std::endl;
}

}  // namespace initialpose_estimation
