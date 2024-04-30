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
#include "bbs3d/random_search.hpp"

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

  // prepare BBS3D
  BBS3D bbs3d;

  // prepare ndt
  // pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr mg_ndt_omp(new pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>());
  // mg_ndt_omp->setResolution(2.0);
  // mg_ndt_omp->setNumThreads(4);
  // mg_ndt_omp->setMaximumIterations(30);
  // mg_ndt_omp->setTransformationEpsilon(0.0);
  // mg_ndt_omp->createVoxelKdtree();

  std::cout << std::fixed;

  Timer timer;

  const Eigen::Matrix4f initial_pose = initial_pose_list.front();
  const std::string& source_pcd = source_pcd_list.front();
  const pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud = load_pcd(source_pcd);
  std::cout << "source_cloud->size(): " << source_cloud->size() << std::endl;

  std::vector<Eigen::Vector3d> src_points;
  double max_norm = 0.0;
  for(const auto& point : source_cloud->points) {
    src_points.emplace_back(point.x, point.y, point.z);
    max_norm = std::max(max_norm, src_points.back().norm());
  }
  timer.start();
  bbs3d.set_src_points(src_points);
  const double milliseconds_src = timer.elapsed_milliseconds();
  std::cout << "set_src_points: " << milliseconds_src << " ms" << std::endl;

  const double kSearchWidth = 10.0;
  const double cloud_width = kSearchWidth + max_norm;

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
  timer.start();
  bbs3d.set_tar_points(target_points, min_level_res, max_level);
  const double milliseconds_tar = timer.elapsed_milliseconds();
  std::cout << "set_tar_points: " << milliseconds_tar << " ms" << std::endl;

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
  timer.start();
  bbs3d.set_trans_search_range(min_xyz, max_xyz);
  const double milliseconds_set_trans_search_range = timer.elapsed_milliseconds();
  std::cout << "set_trans_search_range: " << milliseconds_set_trans_search_range << " ms" << std::endl;

  timer.start();
  bbs3d.localize();
  const double milliseconds_localize = timer.elapsed_milliseconds();
  std::cout << "localize: " << milliseconds_localize << " ms" << std::endl;

  const int best_score = bbs3d.get_best_score();
  std::cout << "best_score: " << best_score << std::endl;
  std::cout << "best_score_percentage: " << bbs3d.get_best_score_percentage() << std::endl;

  const Eigen::Matrix4d global_pose = bbs3d.get_global_pose();

  std::cout << std::fixed;
  std::cout << "gt_pose   = " << initial_pose(0, 3) << ", " << initial_pose(1, 3) << ", " << initial_pose(2, 3) << std::endl;
  std::cout << "pred_pose = " << global_pose(0, 3) << ", " << global_pose(1, 3) << ", " << global_pose(2, 3) << std::endl;
}
