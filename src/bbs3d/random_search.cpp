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

#include "bbs3d/random_search.hpp"

#include <geometry_msgs/msg/vector3.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <random>

namespace random_search {

struct Particle {
  Particle(const geometry_msgs::msg::Pose& a_initial_pose, const geometry_msgs::msg::Pose& a_result_pose, const double a_score, const int a_iteration) : initial_pose(a_initial_pose), result_pose(a_result_pose), score(a_score), iteration(a_iteration) {}
  geometry_msgs::msg::Pose initial_pose;
  geometry_msgs::msg::Pose result_pose;
  double score;
  int iteration;
};

geometry_msgs::msg::Vector3 get_rpy(const geometry_msgs::msg::Pose& pose) {
  geometry_msgs::msg::Vector3 rpy;
  tf2::Quaternion q(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
  tf2::Matrix3x3(q).getRPY(rpy.x, rpy.y, rpy.z);
  return rpy;
}

Eigen::Affine3d pose_to_affine3d(const geometry_msgs::msg::Pose& ros_pose) {
  Eigen::Affine3d eigen_pose;
  tf2::fromMsg(ros_pose, eigen_pose);
  return eigen_pose;
}

Eigen::Matrix4f pose_to_matrix4f(const geometry_msgs::msg::Pose& ros_pose) {
  Eigen::Affine3d eigen_pose_affine = pose_to_affine3d(ros_pose);
  Eigen::Matrix4f eigen_pose_matrix = eigen_pose_affine.matrix().cast<float>();
  return eigen_pose_matrix;
}

geometry_msgs::msg::Pose matrix4f_to_pose(const Eigen::Matrix4f& eigen_pose_matrix) {
  Eigen::Affine3d eigen_pose_affine;
  eigen_pose_affine.matrix() = eigen_pose_matrix.cast<double>();
  geometry_msgs::msg::Pose ros_pose = tf2::toMsg(eigen_pose_affine);
  return ros_pose;
}

geometry_msgs::msg::PoseWithCovarianceStamped random_search(std::shared_ptr<NormalDistributionsTransform> ndt_ptr, const geometry_msgs::msg::PoseWithCovarianceStamped& initial_pose_with_cov, const int64_t particles_num) {
  const geometry_msgs::msg::Vector3 base_rpy = get_rpy(initial_pose_with_cov.pose.pose);
  const Eigen::Map<const Eigen::Matrix<double, 6, 6>> covariance = {initial_pose_with_cov.pose.covariance.data(), 6, 6};
  const double stddev_x = std::sqrt(covariance(0, 0));
  const double stddev_y = std::sqrt(covariance(1, 1));
  const double stddev_z = std::sqrt(covariance(2, 2));
  const double stddev_roll = std::sqrt(covariance(3, 3));
  const double stddev_pitch = std::sqrt(covariance(4, 4));

  std::vector<Particle> particle_array;

  auto output_cloud = std::make_shared<pcl::PointCloud<PointSource>>();

  std::mt19937_64 engine(std::random_device{}());
  std::normal_distribution<double> normal_tx(initial_pose_with_cov.pose.pose.position.x, stddev_x);
  std::normal_distribution<double> normal_ty(initial_pose_with_cov.pose.pose.position.y, stddev_y);
  std::normal_distribution<double> normal_tz(initial_pose_with_cov.pose.pose.position.z, stddev_z);
  std::normal_distribution<double> normal_ar(base_rpy.x, stddev_roll);
  std::normal_distribution<double> normal_ap(base_rpy.y, stddev_pitch);
  std::uniform_real_distribution<double> uniform_ar(-M_PI, M_PI);

  for(int64_t i = 0; i < particles_num; i++) {
    geometry_msgs::msg::Pose initial_pose;
    initial_pose.position.x = normal_tx(engine);
    initial_pose.position.y = normal_ty(engine);
    initial_pose.position.z = normal_tz(engine);
    geometry_msgs::msg::Vector3 init_rpy;
    init_rpy.x = normal_ar(engine);
    init_rpy.y = normal_ap(engine);
    init_rpy.z = uniform_ar(engine);
    tf2::Quaternion tf_quaternion;
    tf_quaternion.setRPY(init_rpy.x, init_rpy.y, init_rpy.z);
    initial_pose.orientation = tf2::toMsg(tf_quaternion);

    const Eigen::Matrix4f initial_pose_matrix = pose_to_matrix4f(initial_pose);
    ndt_ptr->align(*output_cloud, initial_pose_matrix);
    const pclomp::NdtResult ndt_result = ndt_ptr->getResult();

    Particle particle(initial_pose, matrix4f_to_pose(ndt_result.pose), ndt_result.transform_probability, ndt_result.iteration_num);
    particle_array.push_back(particle);
  }

  auto best_particle_ptr = std::max_element(std::begin(particle_array), std::end(particle_array), [](const Particle& lhs, const Particle& rhs) { return lhs.score < rhs.score; });

  geometry_msgs::msg::PoseWithCovarianceStamped result_pose_with_cov_msg;
  result_pose_with_cov_msg.header.stamp = initial_pose_with_cov.header.stamp;
  result_pose_with_cov_msg.header.frame_id = "map";
  result_pose_with_cov_msg.pose.pose = best_particle_ptr->result_pose;

  return result_pose_with_cov_msg;
}

}  // namespace random_search
