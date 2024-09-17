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

#include "timer.hpp"

#include <autoware/ndt_omp/multigrid_pclomp/multigrid_ndt_omp.h>
#include <autoware/ndt_omp/pclomp/gicp_omp.h>
#include <glob.h>
#include <omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/visualization/pcl_visualizer.h>

// must include after including pcl header
// clang-format off
#include "pcd_map_grid_manager.hpp"
#include "util.hpp"
// clang-format on

#include <chrono>
#include <filesystem>
#include <iostream>

int main(int argc, char ** argv)
{
  if (argc != 3) {
    std::cout << "usage: ./regression_test <input_dir> <output_dir>" << std::endl;
    return 0;
  }

  const std::string input_dir = argv[1];
  const std::string output_dir = argv[2];

  // load target pcd
  const std::string target_pcd = input_dir + "/pointcloud_map.pcd";
  const pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud = load_pcd_recursive(target_pcd);

  // prepare sensor_pcd
  const std::string source_pcd_dir = input_dir + "/sensor_pcd/";
  const std::vector<std::string> source_pcd_list = glob(source_pcd_dir);

  // load kinematic_state.csv
  const std::vector<Eigen::Matrix4f> initial_pose_list =
    load_pose_list(input_dir + "/kinematic_state.csv");

  if (initial_pose_list.size() != source_pcd_list.size()) {
    std::cerr << "initial_pose_list.size() != source_pcd_list.size()" << std::endl;
    return 1;
  }
  const auto n_data = static_cast<int64_t>(initial_pose_list.size());

  // prepare ndt
  autoware::ndt_omp::pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::
    Ptr mg_ndt_omp(new autoware::ndt_omp::pclomp::MultiGridNormalDistributionsTransform<
                   pcl::PointXYZ, pcl::PointXYZ>());
  mg_ndt_omp->setResolution(2.0);
  mg_ndt_omp->setNumThreads(4);
  mg_ndt_omp->setMaximumIterations(30);
  mg_ndt_omp->setTransformationEpsilon(0.0);
  mg_ndt_omp->createVoxelKdtree();

  // prepare map grid manager
  MapGridManager map_grid_manager(target_cloud);

  // prepare results
  std::vector<double> elapsed_milliseconds;
  std::vector<double> nvtl_scores;
  std::vector<double> tp_scores;

  std::cout << std::fixed;

  constexpr int update_interval = 10;
  Timer timer;

  // execute align
  for (int64_t i = 0; i < n_data; i++) {
    // get input
    const Eigen::Matrix4f & initial_pose = initial_pose_list[i];
    const std::string & source_pcd = source_pcd_list[i];
    const pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud = load_pcd(source_pcd);
    mg_ndt_omp->setInputSource(source_cloud);

    // update map
    if (i % update_interval == 0) {
      const auto [add_result, remove_result] = map_grid_manager.query(initial_pose);
      std::cout << "add_result.size()=" << std::setw(3) << add_result.size()
                << ", remove_result.size()=" << std::setw(3) << remove_result.size() << ", ";
      for (const auto & [key, cloud] : add_result) {
        mg_ndt_omp->addTarget(cloud, key);
      }
      for (const auto & key : remove_result) {
        mg_ndt_omp->removeTarget(key);
      }
      mg_ndt_omp->createVoxelKdtree();
    }

    // align
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
    timer.start();
    mg_ndt_omp->align(*aligned, initial_pose);
    const autoware::ndt_omp::pclomp::NdtResult ndt_result = mg_ndt_omp->getResult();
    const double elapsed = timer.elapsed_milliseconds();

    const float gain_tp =
      ndt_result.transform_probability - ndt_result.transform_probability_array.front();
    const float gain_nvtl = ndt_result.nearest_voxel_transformation_likelihood -
                            ndt_result.nearest_voxel_transformation_likelihood_array.front();

    // output result
    const double tp = ndt_result.transform_probability;
    const double nvtl = ndt_result.nearest_voxel_transformation_likelihood;
    elapsed_milliseconds.push_back(elapsed);
    nvtl_scores.push_back(nvtl);
    tp_scores.push_back(tp);
    if (i % update_interval == 0) {
      std::cout << "source_cloud->size()=" << std::setw(4) << source_cloud->size()
                << ", time=" << elapsed << " [msec], nvtl=" << nvtl << ", tp = " << tp
                << ", gain_nvtl=" << gain_nvtl << ", gain_tp=" << gain_tp << std::endl;
    }
  }

  // output result
  mkdir(output_dir.c_str(), 0777);
  std::ofstream ofs(output_dir + "/result.csv");
  ofs << "elapsed_milliseconds,nvtl_score,tp_score" << std::endl;
  ofs << std::fixed;
  for (size_t i = 0; i < elapsed_milliseconds.size(); i++) {
    ofs << elapsed_milliseconds[i] << "," << nvtl_scores[i] << "," << tp_scores[i] << std::endl;
  }
}
