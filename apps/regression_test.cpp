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
  pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr mg_ndt_omp(new pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>());
  mg_ndt_omp->setResolution(2.0);
  mg_ndt_omp->setNumThreads(4);
  mg_ndt_omp->setInputTarget(target_cloud);
  mg_ndt_omp->setMaximumIterations(30);
  mg_ndt_omp->setTransformationEpsilon(0.0);
  mg_ndt_omp->createVoxelKdtree();

  // prepare results
  std::vector<double> elapsed_milliseconds;
  std::vector<double> nvtl_scores;
  std::vector<double> tp_scores;

  std::cout << std::fixed;

  // execute align
  for(int64_t i = 0; i < n_data; i++) {
    const Eigen::Matrix4f initial_pose = initial_pose_list[i];
    const std::string& source_pcd = source_pcd_list[i];
    const pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud = load_pcd(source_pcd);
    mg_ndt_omp->setInputSource(source_cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
    auto t1 = std::chrono::system_clock::now();
    mg_ndt_omp->align(*aligned, initial_pose);
    const pclomp::NdtResult ndt_result = mg_ndt_omp->getResult();
    auto t2 = std::chrono::system_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
    const double tp = ndt_result.transform_probability;
    const double nvtl = ndt_result.nearest_voxel_transformation_likelihood;
    elapsed_milliseconds.push_back(elapsed);
    nvtl_scores.push_back(nvtl);
    tp_scores.push_back(tp);
    std::cout << source_pcd << ", num=" << std::setw(4) << source_cloud->size() << " points, time=" << elapsed << " [msec], nvtl=" << nvtl << ", tp = " << tp << std::endl;
  }

  // output result
  mkdir(output_dir.c_str(), 0777);
  std::ofstream ofs(output_dir + "/result.csv");
  ofs << "elapsed_milliseconds,nvtl_score,tp_score" << std::endl;
  ofs << std::fixed;
  for(size_t i = 0; i < elapsed_milliseconds.size(); i++) {
    ofs << elapsed_milliseconds[i] << "," << nvtl_scores[i] << "," << tp_scores[i] << std::endl;
  }
}
