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
#include "estimate_covariance/estimate_covariance.hpp"

std::vector<std::string> glob(const std::string& input_dir) {
  glob_t buffer;
  std::vector<std::string> files;
  glob((input_dir + "/*").c_str(), 0, NULL, &buffer);
  for(size_t i = 0; i < buffer.gl_pathc; i++) {
    files.push_back(buffer.gl_pathv[i]);
  }
  globfree(&buffer);
  std::sort(files.begin(), files.end());
  return files;
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
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  if(pcl::io::loadPCDFile(target_pcd, *target_cloud)) {
    std::cerr << "failed to load " << target_pcd << std::endl;
    return 1;
  }

  // prepare ndt
  std::shared_ptr<pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>> mg_ndt_omp(new pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>());
  mg_ndt_omp->setInputTarget(target_cloud);
  mg_ndt_omp->setResolution(2.0);
  mg_ndt_omp->setNumThreads(4);
  mg_ndt_omp->setMaximumIterations(30);
  mg_ndt_omp->setTransformationEpsilon(0.01);
  mg_ndt_omp->setStepSize(0.1);
  mg_ndt_omp->createVoxelKdtree();

  // prepare sensor_pcd
  const std::string source_pcd_dir = input_dir + "/sensor_pcd/";
  std::vector<std::string> source_pcd_list = glob(source_pcd_dir);

  // prepare results
  std::vector<double> elapsed_milliseconds;
  std::vector<double> scores;

  // load kinematic_state.csv
  /*
  timestamp,pose_x,pose_y,pose_z,quat_w,quat_x,quat_y,quat_z,twist_linear_x,twist_linear_y,twist_linear_z,twist_angular_x,twist_angular_y,twist_angular_z
  63.100010,81377.359702,49916.899866,41.232589,0.953768,0.000494,-0.007336,0.300453,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
  63.133344,81377.359780,49916.899912,41.232735,0.953769,0.000491,-0.007332,0.300452,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
  ...
  */
  std::ifstream ifs(input_dir + "/kinematic_state.csv");
  std::string line;
  std::getline(ifs, line);  // skip header
  std::vector<Eigen::Matrix4f> initial_pose_list;
  while(std::getline(ifs, line)) {
    std::istringstream iss(line);
    std::string token;
    std::vector<std::string> tokens;
    while(std::getline(iss, token, ',')) {
      tokens.push_back(token);
    }
    const double timestamp = std::stod(tokens[0]);
    const double pose_x = std::stod(tokens[1]);
    const double pose_y = std::stod(tokens[2]);
    const double pose_z = std::stod(tokens[3]);
    const double quat_w = std::stod(tokens[4]);
    const double quat_x = std::stod(tokens[5]);
    const double quat_y = std::stod(tokens[6]);
    const double quat_z = std::stod(tokens[7]);
    Eigen::Matrix4f initial_pose = Eigen::Matrix4f::Identity();
    initial_pose.block<3, 3>(0, 0) = Eigen::Quaternionf(quat_w, quat_x, quat_y, quat_z).toRotationMatrix();
    initial_pose.block<3, 1>(0, 3) = Eigen::Vector3f(pose_x, pose_y, pose_z);
    initial_pose_list.push_back(initial_pose);
  }

  if(initial_pose_list.size() != source_pcd_list.size()) {
    std::cerr << "initial_pose_list.size() != source_pcd_list.size()" << std::endl;
    return 1;
  }
  const int64_t n_data = initial_pose_list.size();

  std::cout << std::fixed;

  const std::vector<double> offset_x = {0.0, 0.0, 0.5, -0.5, 1.0, -1.0};
  const std::vector<double> offset_y = {0.5, -0.5, 0.0, 0.0, 0.0, 0.0};

  // output result
  mkdir(output_dir.c_str(), 0777);
  std::ofstream ofs(output_dir + "/result.csv");
  ofs << std::fixed;
  ofs << "index,elapsed_milliseconds,score,x,y,cov_by_la_00,cov_by_la_01,cov_by_la_10,cov_by_la_11,cov_by_mndt_00,cov_by_mndt_01,cov_by_mndt_10,cov_by_mndt_11" << std::endl;

  // execute align
  for(int64_t i = 0; i < n_data; i++) {
    const Eigen::Matrix4f initial_pose = initial_pose_list[i];
    const std::string& source_pcd = source_pcd_list[i];
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if(pcl::io::loadPCDFile(source_pcd, *source_cloud)) {
      std::cerr << "failed to load " << source_pcd << std::endl;
      return 1;
    }
    mg_ndt_omp->setInputSource(source_cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
    auto t1 = std::chrono::system_clock::now();
    mg_ndt_omp->align(*aligned, initial_pose);
    const pclomp::NdtResult ndt_result = mg_ndt_omp->getResult();
    auto t2 = std::chrono::system_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
    const double score = ndt_result.nearest_voxel_transformation_likelihood;
    elapsed_milliseconds.push_back(elapsed);
    scores.push_back(score);
    std::cout << source_pcd << ", num=" << std::setw(4) << source_cloud->size() << " points, time=" << elapsed << " [msec], score=" << score << std::endl;

    // estimate covariance
    const Eigen::Matrix2d cov_by_la = pclomp::estimate_xy_covariance_by_Laplace_approximation(ndt_result);
    const Eigen::Matrix2d cov_by_mndt = pclomp::estimate_xy_covariance_by_multi_ndt(ndt_result, mg_ndt_omp, initial_pose, offset_x, offset_y);

    ofs << i << "," << elapsed << "," << score << "," << initial_pose(0, 3) << "," << initial_pose(1, 3) << "," << cov_by_la(0, 0) << "," << cov_by_la(0, 1) << "," << cov_by_la(1, 0) << "," << cov_by_la(1, 1) << "," << cov_by_mndt(0, 0) << "," << cov_by_mndt(0, 1) << "," << cov_by_mndt(1, 0) << "," << cov_by_mndt(1, 1) << std::endl;
  }
}
