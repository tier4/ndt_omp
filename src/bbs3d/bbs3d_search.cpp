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
  // get pose information
  const Eigen::Matrix4f initial_pose = pose_to_matrix4f(initial_pose_with_cov.pose.pose);
  const Eigen::Map<const Eigen::Matrix<double, 6, 6>> covariance = {initial_pose_with_cov.pose.covariance.data(), 6, 6};
  const double stddev_x = std::sqrt(covariance(0, 0));
  const double stddev_y = std::sqrt(covariance(1, 1));
  const double stddev_z = std::sqrt(covariance(2, 2));
  const double stddev_roll = std::sqrt(covariance(3, 3));
  const double stddev_pitch = std::sqrt(covariance(4, 4));
  const double coeff = 4.0;  // 4 sigma (99.9936%)
  const double search_width_x = coeff * stddev_x;
  const double search_width_y = coeff * stddev_y;
  const double search_width_z = coeff * stddev_z;
  const double search_width_roll = coeff * stddev_roll;
  const double search_width_pitch = coeff * stddev_pitch;

  // calc norm of sensor pcd
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr source_cloud = ndt_ptr->getInputSource();
  float sensor_pcd_max_norm = 0.0;
  for(const auto& point : source_cloud->points) {
    sensor_pcd_max_norm = std::max(sensor_pcd_max_norm, std::hypot(point.x, point.y, point.z));
  }
  const double pcd_width_x = search_width_x + sensor_pcd_max_norm;
  const double pcd_width_y = search_width_y + sensor_pcd_max_norm;

  // prepare BBS3D
  BBS3D bbs3d;

  // set target points
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr target_cloud_const = ndt_ptr->getInputTarget();
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>(*target_cloud_const));
  pcl::PassThrough<pcl::PointXYZ> pass_x;
  pass_x.setInputCloud(target_cloud);
  pass_x.setFilterFieldName("x");
  pass_x.setFilterLimits(initial_pose(0, 3) - pcd_width_x, initial_pose(0, 3) + pcd_width_x);
  pass_x.filter(*target_cloud);
  pcl::PassThrough<pcl::PointXYZ> pass_y;
  pass_y.setInputCloud(target_cloud);
  pass_y.setFilterFieldName("y");
  pass_y.setFilterLimits(initial_pose(1, 3) - pcd_width_y, initial_pose(1, 3) + pcd_width_y);
  pass_y.filter(*target_cloud);
  std::vector<Eigen::Vector3d> target_points;
  for(const auto& point : target_cloud->points) {
    target_points.emplace_back(point.x, point.y, point.z);
  }
  const double min_level_res = 0.5;
  const int max_level = 4;
  bbs3d.set_tar_points(target_points, min_level_res, max_level);

  // set search range
  Eigen::Vector3d min_xyz;
  min_xyz.x() = initial_pose(0, 3) - search_width_x;
  min_xyz.y() = initial_pose(1, 3) - search_width_y;
  min_xyz.z() = initial_pose(2, 3) - search_width_z;
  Eigen::Vector3d max_xyz;
  max_xyz.x() = initial_pose(0, 3) + search_width_x;
  max_xyz.y() = initial_pose(1, 3) + search_width_y;
  max_xyz.z() = initial_pose(2, 3) + search_width_z;
  bbs3d.set_trans_search_range(min_xyz, max_xyz);
  const Eigen::Vector3f base_rpy = initial_pose.block<3, 3>(0, 0).eulerAngles(0, 1, 2);
  Eigen::Vector3d min_rpy;
  min_rpy.x() = base_rpy.x() - search_width_roll;
  min_rpy.y() = base_rpy.y() - search_width_pitch;
  min_rpy.z() = -M_PI;
  Eigen::Vector3d max_rpy;
  max_rpy.x() = base_rpy.x() + search_width_roll;
  max_rpy.y() = base_rpy.y() + search_width_pitch;
  max_rpy.z() = M_PI;
  bbs3d.set_angular_search_range(min_rpy, max_rpy);

  // other settings
  bbs3d.set_score_threshold_percentage(0.25);
  bbs3d.enable_timeout();
  bbs3d.set_timeout_duration_in_msec(500);

  SearchResult result;
  result.score = 0.0;

  for(const double div : {1.0, 2.0, 4.0}) {
    // set curr src points
    const double limit_norm = sensor_pcd_max_norm / div;
    std::vector<Eigen::Vector3d> src_points;
    for(const auto& point : source_cloud->points) {
      const double norm = std::hypot(point.x, point.y, point.z);
      if(norm > limit_norm) {
        continue;
      }
      src_points.emplace_back(point.x, point.y, point.z);
    }
    bbs3d.set_src_points(src_points);

    // search
    bbs3d.localize_by_chokudai_search();
    const Eigen::Matrix4d global_pose = bbs3d.get_global_pose();

    // align
    ndt_ptr->align(*target_cloud, global_pose.cast<float>());
    const pclomp::NdtResult ndt_result = ndt_ptr->getResult();
    const Eigen::Matrix4f ndt_result_pose = ndt_result.pose;
    if(ndt_result.transform_probability > result.score) {
      result.pose_with_cov.pose.pose = matrix4f_to_pose(ndt_result_pose);
      result.score = ndt_result.transform_probability;
    }
  }

  return result;
}

}  // namespace initialpose_estimation
