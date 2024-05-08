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

#include "util.hpp"
#include "pcd_map_grid_manager.hpp"
#include "timer.hpp"
#include "particle_filter/particle_filter.hpp"

int main(int argc, char** argv) {
  if(argc != 3) {
    std::cout << "usage: ./exec_particle_filter <input_dir> <output_dir>" << std::endl;
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
  pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr mg_ndt_omp(new pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>());
  mg_ndt_omp->setResolution(2.0);
  mg_ndt_omp->setNumThreads(4);
  mg_ndt_omp->setMaximumIterations(3);
  mg_ndt_omp->setTransformationEpsilon(0.0);
  mg_ndt_omp->createVoxelKdtree();

  std::mutex mutex_ndt;

  // prepare map grid manager
  MapGridManager map_grid_manager(target_cloud);

  // prepare particle filter
  particle_filter::ParticleFilterParams params;
  params.num_particles = 100;
  params.initial_pose = initial_pose_list[0];
  params.covariance_diagonal = {1.0f, 1.0f, 0.01f, 0.01f, 0.01f, 10.0f};
  params.score_function = [&](const Eigen::Matrix4f& pose) {
    std::lock_guard<std::mutex> lock(mutex_ndt);
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_pcd(new pcl::PointCloud<pcl::PointXYZ>());
    mg_ndt_omp->align(*aligned_pcd, pose);
    const pclomp::NdtResult ndt_result = mg_ndt_omp->getResult();
    // return ndt_result.nearest_voxel_transformation_likelihood;
    return ndt_result.transform_probability;
  };
  particle_filter::ParticleFilter particle_filter(params);

  std::cout << std::fixed;

  constexpr int update_interval = 10;
  Timer timer;

  // output result
  mkdir(output_dir.c_str(), 0777);
  std::ofstream ofs_elapsed(output_dir + "/elapsed.csv");
  ofs_elapsed << "elapsed_predict,elapsed_update,elapsed_resample" << std::endl;
  ofs_elapsed << std::fixed;
  mkdir((output_dir + "/particles").c_str(), 0777);

  // execute align
  for(int64_t i = 1; i < n_data; i++) {
    // get input
    const Eigen::Matrix4f curr_pose = initial_pose_list[i];
    const Eigen::Matrix4f delta_pose = initial_pose_list[i - 1].inverse() * initial_pose_list[i];
    const std::string& source_pcd = source_pcd_list[i];
    const pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud = load_pcd(source_pcd);
    mg_ndt_omp->setInputSource(source_cloud);

    // update map
    if(i == 1 || i % update_interval == 0) {
      const auto [add_result, remove_result] = map_grid_manager.query(curr_pose);
      std::cout << "add_result.size()=" << std::setw(3) << add_result.size() << ", remove_result.size()=" << std::setw(3) << remove_result.size() << ", ";
      for(const auto& [key, cloud] : add_result) {
        mg_ndt_omp->addTarget(cloud, key);
      }
      for(const auto& key : remove_result) {
        mg_ndt_omp->removeTarget(key);
      }
      mg_ndt_omp->createVoxelKdtree();
    }

    // predict
    timer.start();
    particle_filter.predict(delta_pose);
    const double elapsed_predict = timer.elapsed_milliseconds();

    // update
    timer.start();
    particle_filter.update();
    const double elapsed_update = timer.elapsed_milliseconds();

    // resample
    timer.start();
    particle_filter.resample();
    const double elapsed_resample = timer.elapsed_milliseconds();

    const float ess = particle_filter.effective_sample_size();

    std::cout << "elapsed_predict=" << elapsed_predict << ", elapsed_update=" << elapsed_update << ", elapsed_resample=" << elapsed_resample << ", ess=" << ess << std::endl;
    ofs_elapsed << elapsed_predict << "," << elapsed_update << "," << elapsed_resample << std::endl;

    // output particles
    std::vector<particle_filter::Particle> particles = particle_filter.get_particles();
    std::sort(particles.begin(), particles.end(), [](const particle_filter::Particle& a, const particle_filter::Particle& b) { return a.weight > b.weight; });

    const std::string particles_path = (std::stringstream() << output_dir << "/particles/" << std::setw(8) << std::setfill('0') << i << ".csv").str();
    std::ofstream ofs_particles(particles_path);
    ofs_particles << "x,y,z,roll,pitch,yaw,score,weight" << std::endl;
    ofs_particles << std::fixed;
    const int64_t size = particles.size();
    for(int64_t j = 0; j < size; j++) {
      const particle_filter::Particle& particle = particles[j];
      const Eigen::Vector3f rpy = particle.pose.block<3, 3>(0, 0).eulerAngles(0, 1, 2);
      ofs_particles << particle.pose(0, 3) << ",";
      ofs_particles << particle.pose(1, 3) << ",";
      ofs_particles << particle.pose(2, 3) << ",";
      ofs_particles << rpy(0) << ",";
      ofs_particles << rpy(1) << ",";
      ofs_particles << rpy(2) << ",";
      ofs_particles << particle.score << ",";
      ofs_particles << particle.weight << std::endl;
    }
  }
}
