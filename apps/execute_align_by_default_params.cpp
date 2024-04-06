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

std::ostream& operator<<(std::ostream& ost, const Eigen::Matrix4f& pose) {
  for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 4; j++) {
      ost << pose(i, j);
      if(i != 3 || j != 3) {
        ost << ",";
      }
    }
  }
  return ost;
}

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

  // prepare aligner
  pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr ndt_omp(new pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>());
  ndt_omp->setResolution(2.0);
  ndt_omp->setNumThreads(4);
  ndt_omp->setMaximumIterations(30);
  ndt_omp->setTransformationEpsilon(0.01);
  ndt_omp->setStepSize(0.1);
  ndt_omp->setInputTarget(target_cloud);
  ndt_omp->createVoxelKdtree();

  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>::Ptr fast_gicp(new fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ>());
  fast_gicp->setNumThreads(4);
  fast_gicp->setMaximumIterations(30);
  fast_gicp->setTransformationEpsilon(0.01);
  fast_gicp->setInputTarget(target_cloud);

  Timer timer;

  mkdir(output_dir.c_str(), 0777);
  std::ofstream ofs(output_dir + "/result.csv");
  ofs << std::fixed;

  const std::vector<std::string> methods = {"ndt_omp", "fast_gicp"};
  for(const std::string& method : methods) {
    ofs << method << "_elapsed_msec,";
    for(int i = 0; i < 4; i++) {
      for(int j = 0; j < 4; j++) {
        ofs << method << "_pose" << i << j;
        if(i != 3 || j != 3) {
          ofs << ",";
        }
      }
    }
    if (method == methods.back()) {
      ofs << std::endl;
    } else {
      ofs << ",";
    }
  }

  // execute align
  for(int64_t i = 0; i < n_data; i++) {
    // get input
    const Eigen::Matrix4f initial_pose = initial_pose_list[i];
    const std::string& source_pcd = source_pcd_list[i];
    const pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud = load_pcd(source_pcd);
    ndt_omp->setInputSource(source_cloud);
    fast_gicp->setInputSource(source_cloud);

    // align
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());

    // ndt_omp
    timer.start();
    ndt_omp->align(*aligned, initial_pose);
    const double elapsed_ndt_omp = timer.elapsed_milliseconds();
    const pclomp::NdtResult result_ndt_omp_struct = ndt_omp->getResult();
    const Eigen::Matrix4f result_ndt_omp = result_ndt_omp_struct.transformation_array.back();
    ofs << elapsed_ndt_omp << "," << result_ndt_omp << ",";

    // fast_gicp
    timer.start();
    fast_gicp->align(*aligned, initial_pose);
    const double elapsed_fast_gicp = timer.elapsed_milliseconds();
    const Eigen::Matrix4f result_fast_gicp = fast_gicp->getFinalTransformation();
    ofs << elapsed_fast_gicp << "," << result_fast_gicp;

    ofs << std::endl;

    std::cout << i << "\r" << std::flush;
  }
}
