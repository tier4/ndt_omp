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

#ifndef INITIAL_POSE_ESTIMATION_HPP
#define INITIAL_POSE_ESTIMATION_HPP

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <pcl/point_types.h>

#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "multigrid_pclomp/multigrid_ndt_omp.h"

namespace initialpose_estimation {
using PointSource = pcl::PointXYZ;
using PointTarget = pcl::PointXYZ;
using NormalDistributionsTransform = pclomp::MultiGridNormalDistributionsTransform<PointSource, PointTarget>;

struct SearchResult {
  geometry_msgs::msg::PoseWithCovarianceStamped pose_with_cov;
  double score;
  int64_t search_count;
};

class Timer {
public:
  void start() {
    start_time_ = std::chrono::steady_clock::now();
  }

  int64_t elapsed_milli_seconds() const {
    auto elapsed = std::chrono::steady_clock::now() - start_time_;
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  }

private:
  std::chrono::steady_clock::time_point start_time_;
};

// main functions
SearchResult random_search(std::shared_ptr<NormalDistributionsTransform> ndt_ptr, const geometry_msgs::msg::PoseWithCovarianceStamped& initial_pose_with_cov, const int64_t limit_msec);
SearchResult bbs3d_search(std::shared_ptr<NormalDistributionsTransform> ndt_ptr, const geometry_msgs::msg::PoseWithCovarianceStamped& initial_pose_with_cov, const int64_t limit_msec);
SearchResult tpe_search(std::shared_ptr<NormalDistributionsTransform> ndt_ptr, const geometry_msgs::msg::PoseWithCovarianceStamped& initial_pose_with_cov, const int64_t limit_msec);

// utils
inline Eigen::Affine3d pose_to_affine3d(const geometry_msgs::msg::Pose& ros_pose) {
  Eigen::Affine3d eigen_pose;
  tf2::fromMsg(ros_pose, eigen_pose);
  return eigen_pose;
}

inline Eigen::Matrix4f pose_to_matrix4f(const geometry_msgs::msg::Pose& ros_pose) {
  Eigen::Affine3d eigen_pose_affine = pose_to_affine3d(ros_pose);
  Eigen::Matrix4f eigen_pose_matrix = eigen_pose_affine.matrix().cast<float>();
  return eigen_pose_matrix;
}

inline geometry_msgs::msg::Pose matrix4f_to_pose(const Eigen::Matrix4f& eigen_pose_matrix) {
  Eigen::Affine3d eigen_pose_affine;
  eigen_pose_affine.matrix() = eigen_pose_matrix.cast<double>();
  geometry_msgs::msg::Pose ros_pose = tf2::toMsg(eigen_pose_affine);
  return ros_pose;
}

inline geometry_msgs::msg::Vector3 quaternion_to_rpy(const geometry_msgs::msg::Quaternion& quat) {
  geometry_msgs::msg::Vector3 rpy;
  tf2::Quaternion q(quat.x, quat.y, quat.z, quat.w);
  tf2::Matrix3x3(q).getRPY(rpy.x, rpy.y, rpy.z);
  return rpy;
}

inline geometry_msgs::msg::Quaternion rpy_to_quaternion(const geometry_msgs::msg::Vector3& rpy) {
  tf2::Quaternion tf_quaternion;
  tf_quaternion.setRPY(rpy.x, rpy.y, rpy.z);
  return tf2::toMsg(tf_quaternion);
}

}  // namespace initialpose_estimation

#endif  // INITIAL_POSE_ESTIMATION_HPP
