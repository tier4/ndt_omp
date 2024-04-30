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

// main functions
geometry_msgs::msg::PoseWithCovarianceStamped random_search(std::shared_ptr<NormalDistributionsTransform> ndt_ptr, const geometry_msgs::msg::PoseWithCovarianceStamped& initial_pose_with_cov, const int64_t particles_num);
geometry_msgs::msg::PoseWithCovarianceStamped bbs3d_search(std::shared_ptr<NormalDistributionsTransform> ndt_ptr, const geometry_msgs::msg::PoseWithCovarianceStamped& initial_pose_with_cov, const int64_t particles_num);

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

}  // namespace initialpose_estimation

#endif  // INITIAL_POSE_ESTIMATION_HPP
