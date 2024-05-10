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
#include "bbs3d/tree_structured_parzen_estimator.hpp"
#include <boost/math/special_functions/erf.hpp>

#include <fstream>

namespace initialpose_estimation {

struct Particle {
  Particle(const geometry_msgs::msg::Pose& a_initial_pose, const geometry_msgs::msg::Pose& a_result_pose, const double a_score, const int a_iteration) : initial_pose(a_initial_pose), result_pose(a_result_pose), score(a_score), iteration(a_iteration) {}
  geometry_msgs::msg::Pose initial_pose;
  geometry_msgs::msg::Pose result_pose;
  double score;
  int iteration;
};

SearchResult tpe_search(std::shared_ptr<NormalDistributionsTransform> ndt_ptr, const geometry_msgs::msg::PoseWithCovarianceStamped& initial_pose_with_cov, const int64_t limit_msec) {
  Timer timer;
  timer.start();

  const geometry_msgs::msg::Vector3 base_rpy = quaternion_to_rpy(initial_pose_with_cov.pose.pose.orientation);
  const Eigen::Matrix4f initial_pose = pose_to_matrix4f(initial_pose_with_cov.pose.pose);
  const Eigen::Map<const Eigen::Matrix<double, 6, 6>> covariance = {initial_pose_with_cov.pose.covariance.data(), 6, 6};
  const double stddev_x = std::sqrt(covariance(0, 0));
  const double stddev_y = std::sqrt(covariance(1, 1));
  const double stddev_z = std::sqrt(covariance(2, 2));
  const double stddev_roll = std::sqrt(covariance(3, 3));
  const double stddev_pitch = std::sqrt(covariance(4, 4));

  const std::vector<double> sample_mean{initial_pose_with_cov.pose.pose.position.x, initial_pose_with_cov.pose.pose.position.y, initial_pose_with_cov.pose.pose.position.z, base_rpy.x, base_rpy.y};
  const std::vector<double> sample_stddev{stddev_x, stddev_y, stddev_z, stddev_roll, stddev_pitch};

  // Optimizing (x, y, z, roll, pitch, yaw) 6 dimensions.
  TreeStructuredParzenEstimator tpe(TreeStructuredParzenEstimator::Direction::MAXIMIZE, 20, sample_mean, sample_stddev);

  std::vector<Particle> particle_array;
  auto output_cloud = std::make_shared<pcl::PointCloud<PointSource>>();

  std::ofstream ofs("tpe_search.csv");
  ofs << std::fixed;
  ofs << "index,trans_x,trans_y,trans_z,angle_x,angle_y,angle_z,score" << std::endl;

  for(int64_t i = 0; timer.elapsed_milli_seconds() < limit_msec; i++) {
    const TreeStructuredParzenEstimator::Input input = tpe.get_next_input();

    geometry_msgs::msg::Pose initial_pose;
    initial_pose.position.x = input[0];
    initial_pose.position.y = input[1];
    initial_pose.position.z = input[2];
    geometry_msgs::msg::Vector3 init_rpy;
    init_rpy.x = input[3];
    init_rpy.y = input[4];
    init_rpy.z = input[5];
    tf2::Quaternion tf_quaternion;
    tf_quaternion.setRPY(init_rpy.x, init_rpy.y, init_rpy.z);
    initial_pose.orientation = tf2::toMsg(tf_quaternion);

    const Eigen::Matrix4f initial_pose_matrix = pose_to_matrix4f(initial_pose);
    ndt_ptr->align(*output_cloud, initial_pose_matrix);
    const pclomp::NdtResult ndt_result = ndt_ptr->getResult();

    Particle particle(initial_pose, matrix4f_to_pose(ndt_result.pose), ndt_result.nearest_voxel_transformation_likelihood, ndt_result.iteration_num);
    particle_array.push_back(particle);

    const geometry_msgs::msg::Pose pose = matrix4f_to_pose(ndt_result.pose);
    const geometry_msgs::msg::Vector3 rpy = quaternion_to_rpy(pose.orientation);

    TreeStructuredParzenEstimator::Input result(6);
    result[0] = pose.position.x;
    result[1] = pose.position.y;
    result[2] = pose.position.z;
    result[3] = rpy.x;
    result[4] = rpy.y;
    result[5] = rpy.z;
    tpe.add_trial(TreeStructuredParzenEstimator::Trial{result, ndt_result.nearest_voxel_transformation_likelihood});

    ofs << i << "," << pose.position.x << "," << pose.position.y << "," << pose.position.z << "," << rpy.x << "," << rpy.y << "," << rpy.z << "," << ndt_result.nearest_voxel_transformation_likelihood << std::endl;
  }

  auto best_particle_ptr = std::max_element(std::begin(particle_array), std::end(particle_array), [](const Particle& lhs, const Particle& rhs) { return lhs.score < rhs.score; });

  geometry_msgs::msg::PoseWithCovarianceStamped result_pose_with_cov_msg;
  result_pose_with_cov_msg.header.stamp = initial_pose_with_cov.header.stamp;
  result_pose_with_cov_msg.pose.pose = best_particle_ptr->result_pose;

  SearchResult search_result;
  search_result.pose_with_cov = result_pose_with_cov_msg;
  search_result.score = best_particle_ptr->score;
  search_result.search_count = particle_array.size();
  return search_result;
}

}  // namespace initialpose_estimation
