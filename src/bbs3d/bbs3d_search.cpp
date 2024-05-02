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

  BBS3D bbs3d;

  std::vector<Eigen::Vector3d> src_points;
  double max_norm = 0.0;
  const double kLimitNorm = 20.0;
  for(const auto& point : source_cloud->points) {
    const double norm = std::hypot(point.x, point.y, point.z);
    if(norm > kLimitNorm) {
      continue;
    }
    src_points.emplace_back(point.x, point.y, point.z);
    max_norm = std::max(max_norm, src_points.back().norm());
  }
  bbs3d.set_src_points(src_points);

  const double kSearchWidth = 5.0;
  const double cloud_width = kSearchWidth + max_norm;

  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr target_cloud_const = ndt_ptr->getInputTarget();
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>(*target_cloud_const));

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

  std::vector<Eigen::Vector3d> target_points;
  for(const auto& point : target_cloud->points) {
    target_points.emplace_back(point.x, point.y, point.z);
  }

  const double min_level_res = 0.5;
  const int max_level = 4;
  bbs3d.set_tar_points(target_points, min_level_res, max_level);

  // other settings
  bbs3d.set_score_threshold_percentage(0.25);
  bbs3d.enable_timeout();

  // set search range
  Eigen::Vector3d min_xyz = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
  Eigen::Vector3d max_xyz = Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());
  for(const auto& point : target_points) {
    min_xyz = min_xyz.cwiseMin(point);
    max_xyz = max_xyz.cwiseMax(point);
  }
  min_xyz.x() = initial_pose(0, 3) - kSearchWidth;
  min_xyz.y() = initial_pose(1, 3) - kSearchWidth;
  max_xyz.x() = initial_pose(0, 3) + kSearchWidth;
  max_xyz.y() = initial_pose(1, 3) + kSearchWidth;
  bbs3d.set_trans_search_range(min_xyz, max_xyz);

  bbs3d.set_timeout_duration_in_msec(1000);

  bbs3d.localize_by_chokudai_search();

  const int best_score = bbs3d.get_best_score();

  const Eigen::Matrix4d global_pose = bbs3d.get_global_pose();

  ndt_ptr->align(*target_cloud, global_pose.cast<float>());
  const pclomp::NdtResult ndt_result = ndt_ptr->getResult();
  const Eigen::Matrix4f ndt_result_pose = ndt_result.pose;

  SearchResult result;
  result.pose_with_cov.pose.pose = matrix4f_to_pose(ndt_result_pose);
  result.score = ndt_result.transform_probability;
  return result;
}

}  // namespace initialpose_estimation
