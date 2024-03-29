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
#include <omp.h>
#include <glob.h>
#include <filesystem>

#include <pclomp/gicp_omp.h>
#include <multigrid_pclomp/multigrid_ndt_omp.h>
#include "fast_gicp/gicp/fast_vgicp.hpp"
#include "fast_gicp/gicp/fast_gicp.hpp"

#include "util.hpp"
#include "pcd_map_grid_manager.hpp"
#include "timer.hpp"

int main(int argc, char** argv) {
  if(argc != 3) {
    std::cout << "usage: ./regression_test <input_dir> <output_dir>" << std::endl;
    return 0;
  }

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
  // pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr aligner(new pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>());
  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>::Ptr aligner(new fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>());
  // aligner->setResolution(2.0);
  aligner->setNumThreads(4);
  aligner->setMaximumIterations(30);
  aligner->setTransformationEpsilon(0.01);
  // aligner->setStepSize(0.1);
  // aligner->createVoxelKdtree();
  aligner->setInputTarget(target_cloud);

  // prepare map grid manager
  MapGridManager map_grid_manager(target_cloud);

  // prepare results
  std::vector<int> iteration_nums;
  std::vector<double> elapsed_milliseconds;
  std::vector<double> nvtl_scores;
  std::vector<double> tp_scores;

  std::cout << std::fixed;

  constexpr int update_interval = 10;
  Timer timer;

  // execute align
  for(int64_t i = 0; i < n_data; i++) {
    // get input
    const Eigen::Matrix4f initial_pose = initial_pose_list[i];
    const std::string& source_pcd = source_pcd_list[i];
    const pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud = load_pcd(source_pcd);
    aligner->setInputSource(source_cloud);

    // align
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
    timer.start();
    aligner->align(*aligned, initial_pose);
    const double elapsed = timer.elapsed_milliseconds();

    // output result
    const int iteration_num = aligner->getNrIterations();
    const double tp = 0.0;
    const double nvtl = aligner->getFitnessScore();
    iteration_nums.push_back(iteration_num);
    elapsed_milliseconds.push_back(elapsed);
    nvtl_scores.push_back(nvtl);
    tp_scores.push_back(tp);
    if(i % update_interval == 0) {
      std::cout << "source_cloud->size()=" << std::setw(4) << source_cloud->size() << ", iteration_num=" << iteration_num << ", time=" << elapsed << " [msec], nvtl=" << nvtl << ", tp = " << tp << std::endl;
    }
  }

  // output result
  mkdir(output_dir.c_str(), 0777);
  std::ofstream ofs(output_dir + "/result.csv");
  ofs << "iteration_num,elapsed_milliseconds,nvtl_score,tp_score" << std::endl;
  ofs << std::fixed;
  for(size_t i = 0; i < elapsed_milliseconds.size(); i++) {
    ofs << iteration_nums[i] << "," << elapsed_milliseconds[i] << "," << nvtl_scores[i] << "," << tp_scores[i] << std::endl;
  }
}
