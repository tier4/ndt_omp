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

#include <iostream>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>
#include <omp.h>
#include <glob.h>
#include <filesystem>

#include <pclomp/gicp_omp.h>
#include <multigrid_pclomp/multigrid_ndt_omp.h>

#include "util.hpp"
#include "pcd_map_grid_manager.hpp"
#include "timer.hpp"

#include "bbs3d/bbs3d.hpp"
#include "bbs3d/initialpose_estimation.hpp"

int main(int argc, char** argv) {
  if(argc != 3) {
    std::cout << "usage: ./regression_test <input_dir> <output_dir>" << std::endl;
    return 0;
  }

  std::cout << "start initialpose_estimation" << std::endl;

  const std::string input_dir = argv[1];
  const std::string output_dir = argv[2];

  // load target pcd
  const std::string target_pcd = input_dir + "/pointcloud_map.pcd";
  const pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud = load_pcd(target_pcd);

  // prepare sensor_pcd
  const std::string source_pcd_dir = input_dir + "/sensor_pcd/";
  const std::vector<std::string> source_pcd_list = glob(source_pcd_dir);

  // load kinematic_state.csv
  const std::vector<Eigen::Matrix4f> initial_pose_list = load_pose_list(input_dir + "/kinematic_state.csv");

  if(initial_pose_list.size() != source_pcd_list.size()) {
    std::cerr << "initial_pose_list.size() != source_pcd_list.size()" << std::endl;
    return 1;
  }
  const int64_t n_data = initial_pose_list.size();

  // prepare ndt
  using NDT = pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>;
  NDT::Ptr mg_ndt_omp(new NDT());
  mg_ndt_omp->setResolution(2.0);
  mg_ndt_omp->setNumThreads(4);
  mg_ndt_omp->setMaximumIterations(30);
  mg_ndt_omp->setTransformationEpsilon(0.01);
  mg_ndt_omp->createVoxelKdtree();

  std::cout << std::fixed;

  const Eigen::Matrix4f initial_pose = initial_pose_list.front();
  const std::string& source_pcd = source_pcd_list.front();
  const pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud = load_pcd(source_pcd);
  mg_ndt_omp->setInputSource(source_cloud);

  // filter target_cloud to avoid overflow in creating voxel grid
  pcl::PassThrough<pcl::PointXYZ> pass_x;
  pass_x.setInputCloud(target_cloud);
  pass_x.setFilterFieldName("x");
  pass_x.setFilterLimits(initial_pose(0, 3) - 200, initial_pose(0, 3) + 200);
  pass_x.filter(*target_cloud);
  pcl::PassThrough<pcl::PointXYZ> pass_y;
  pass_y.setInputCloud(target_cloud);
  pass_y.setFilterFieldName("y");
  pass_y.setFilterLimits(initial_pose(1, 3) - 200, initial_pose(1, 3) + 200);
  pass_y.filter(*target_cloud);
  mg_ndt_omp->setInputTarget(target_cloud);

  const double stddev_x = 1.0;
  const double stddev_y = 1.0;

  geometry_msgs::msg::PoseWithCovarianceStamped base_pose;
  base_pose.pose.pose = initialpose_estimation::matrix4f_to_pose(initial_pose);
  base_pose.pose.covariance[0] = stddev_x * stddev_x;  // x
  base_pose.pose.covariance[7] = stddev_y * stddev_y;  // y
  base_pose.pose.covariance[14] = 0.01;                // z
  base_pose.pose.covariance[21] = 0.01;                // roll
  base_pose.pose.covariance[28] = 0.01;                // pitch
  base_pose.pose.covariance[35] = 10.0;                // yaw
  const geometry_msgs::msg::Vector3 base_rpy = initialpose_estimation::quaternion_to_rpy(base_pose.pose.pose.orientation);

  std::filesystem::create_directories(output_dir);
  std::ofstream ofs(output_dir + "/initialpose_estimation.csv");
  ofs << std::fixed;
  ofs << "id,time_msec,score,search_count,initial_trans_x,initial_trans_y,initial_tran_z,initial_angle_x,initial_angle_y,initial_angle_z,result_trans_x,result_trans_y,result_trans_z,result_angle_x,result_angle_y,result_angle_z" << std::endl;

  const double x_unit = stddev_x / 1.0;
  const double y_unit = stddev_y / 1.0;
  const double yaw_unit = 2 * M_PI / 4.0;
  Timer timer;

  int64_t count = 0;

  for(int64_t x_offset = -2; x_offset <= +2; x_offset++) {
    for(int64_t y_offset = -2; y_offset <= +2; y_offset++) {
      for(int64_t yaw_offset = 0; yaw_offset < 4; yaw_offset++) {
        geometry_msgs::msg::PoseWithCovarianceStamped curr_initial_pose = base_pose;
        curr_initial_pose.pose.pose.position.x += x_offset * x_unit;
        curr_initial_pose.pose.pose.position.y += y_offset * y_unit;
        geometry_msgs::msg::Vector3 curr_rpy = base_rpy;
        curr_rpy.z += yaw_offset * yaw_unit;
        curr_initial_pose.pose.pose.orientation = initialpose_estimation::rpy_to_quaternion(curr_rpy);

        timer.start();
        const int64_t limit_msec = 1000;
        const initialpose_estimation::SearchResult result = initialpose_estimation::random_search(mg_ndt_omp, curr_initial_pose, limit_msec);
        // const initialpose_estimation::SearchResult result = initialpose_estimation::bbs3d_search(mg_ndt_omp, curr_initial_pose, limit_msec);
        const double elapsed_time = timer.elapsed_milliseconds();

        const geometry_msgs::msg::PoseWithCovarianceStamped result_pose = result.pose_with_cov;
        const geometry_msgs::msg::Vector3 result_rpy = initialpose_estimation::quaternion_to_rpy(result_pose.pose.pose.orientation);
        const double score = result.score;

        ofs << count << ",";
        ofs << elapsed_time << ",";
        ofs << score << ",";
        ofs << result.search_count << ",";
        ofs << curr_initial_pose.pose.pose.position.x << ",";
        ofs << curr_initial_pose.pose.pose.position.y << ",";
        ofs << curr_initial_pose.pose.pose.position.z << ",";
        ofs << curr_rpy.x << ",";
        ofs << curr_rpy.y << ",";
        ofs << curr_rpy.z << ",";
        ofs << result_pose.pose.pose.position.x << ",";
        ofs << result_pose.pose.pose.position.y << ",";
        ofs << result_pose.pose.pose.position.z << ",";
        ofs << result_rpy.x << ",";
        ofs << result_rpy.y << ",";
        ofs << result_rpy.z << std::endl;
        count++;
        std::cout << count << "\r" << std::flush;
      }
    }
  }
}
