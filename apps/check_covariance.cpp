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

  const std::vector<double> offset_x = {0.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.0, 2.0, -2.0};
  const std::vector<double> offset_y = {0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0};

  // output result
  std::filesystem::create_directories(output_dir);
  std::ofstream ofs(output_dir + "/result.csv");
  ofs << std::fixed;
  ofs << "index,score,initial_x,initial_y,result_x,result_y,elapsed_la,cov_by_la_00,cov_by_la_01,cov_by_la_10,cov_by_la_11,elapsed_mndt,cov_by_mndt_00,cov_by_mndt_01,cov_by_mndt_10,cov_by_mndt_11,elapsed_mndt_score,cov_by_mndt_score_00,cov_by_mndt_score_01,cov_by_mndt_score_10,cov_by_mndt_score_11" << std::endl;

  const std::string multi_ndt_dir = output_dir + "/multi_ndt";
  std::filesystem::create_directories(multi_ndt_dir);
  const std::string multi_ndt_score_dir = output_dir + "/multi_ndt_score";
  std::filesystem::create_directories(multi_ndt_score_dir);

  auto t1 = std::chrono::system_clock::now();
  auto t2 = std::chrono::system_clock::now();

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
    mg_ndt_omp->align(*aligned, initial_pose);
    const pclomp::NdtResult ndt_result = mg_ndt_omp->getResult();
    const double score = ndt_result.nearest_voxel_transformation_likelihood;
    std::cout << source_pcd << ", num=" << std::setw(4) << source_cloud->size() << " points, score=" << score << std::endl;

    const std::vector<Eigen::Matrix4f> poses_to_search = pclomp::propose_poses_to_search(ndt_result.hessian, ndt_result.pose, offset_x, offset_y);

    // estimate covariance
    // (1) Laplace approximation
    t1 = std::chrono::system_clock::now();
    const Eigen::Matrix2d cov_by_la = pclomp::estimate_xy_covariance_by_Laplace_approximation(ndt_result.hessian);
    t2 = std::chrono::system_clock::now();
    const auto elapsed_la = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;

    // (2) Multi NDT
    t1 = std::chrono::system_clock::now();
    const pclomp::ResultOfMultiNdtCovarianceEstimation result_of_mndt = pclomp::estimate_xy_covariance_by_multi_ndt(ndt_result, mg_ndt_omp, poses_to_search);
    const Eigen::Matrix2d cov_by_mndt = result_of_mndt.covariance;
    t2 = std::chrono::system_clock::now();
    const auto elapsed_mndt = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;

    // (3) Multi NDT with score
    t1 = std::chrono::system_clock::now();
    const pclomp::ResultOfMultiNdtCovarianceEstimation result_of_mndt_score = pclomp::estimate_xy_covariance_by_multi_ndt_score(ndt_result, mg_ndt_omp, poses_to_search);
    const Eigen::Matrix2d cov_by_mndt_score = result_of_mndt_score.covariance;
    t2 = std::chrono::system_clock::now();
    const auto elapsed_mndt_score = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;

    // output result
    const auto result_x = ndt_result.pose(0, 3);
    const auto result_y = ndt_result.pose(1, 3);
    ofs << i << "," << score << "," << initial_pose(0, 3) << "," << initial_pose(1, 3) << "," << result_x << "," << result_y;
    ofs << "," << elapsed_la << "," << cov_by_la(0, 0) << "," << cov_by_la(0, 1) << "," << cov_by_la(1, 0) << "," << cov_by_la(1, 1);
    ofs << "," << elapsed_mndt << "," << cov_by_mndt(0, 0) << "," << cov_by_mndt(0, 1) << "," << cov_by_mndt(1, 0) << "," << cov_by_mndt(1, 1);
    ofs << "," << elapsed_mndt_score << "," << cov_by_mndt_score(0, 0) << "," << cov_by_mndt_score(0, 1) << "," << cov_by_mndt_score(1, 0) << "," << cov_by_mndt_score(1, 1) << std::endl;

    std::stringstream filename_ss;
    filename_ss << std::setw(8) << std::setfill('0') << i << ".csv";

    // output multi ndt result
    std::ofstream ofs_mndt(multi_ndt_dir + "/" + filename_ss.str());
    const int n_mndt = result_of_mndt.ndt_results.size();
    ofs_mndt << "index,score,initial_x,initial_y,result_x,result_y" << std::endl;
    ofs_mndt << std::fixed;
    for(int j = 0; j < n_mndt; j++) {
      const pclomp::NdtResult& ndt_result = result_of_mndt.ndt_results[j];
      const auto nvtl = ndt_result.nearest_voxel_transformation_likelihood;
      const auto initial_x = result_of_mndt.ndt_initial_poses[j](0, 3);
      const auto initial_y = result_of_mndt.ndt_initial_poses[j](1, 3);
      const auto result_x = ndt_result.pose(0, 3);
      const auto result_y = ndt_result.pose(1, 3);
      ofs_mndt << j << "," << nvtl << "," << initial_x << "," << initial_y << "," << result_x << "," << result_y << std::endl;
    }

    // output multi ndt score result
    std::ofstream ofs_mndt_score(multi_ndt_score_dir + "/" + filename_ss.str());
    const int n_mndt_score = result_of_mndt_score.ndt_results.size();
    ofs_mndt_score << "index,score,initial_x,initial_y,result_x,result_y" << std::endl;
    ofs_mndt_score << std::fixed;
    for(int j = 0; j < n_mndt_score; j++) {
      const pclomp::NdtResult& ndt_result = result_of_mndt_score.ndt_results[j];
      const auto nvtl = ndt_result.nearest_voxel_transformation_likelihood;
      const auto initial_x = result_of_mndt_score.ndt_initial_poses[j](0, 3);
      const auto initial_y = result_of_mndt_score.ndt_initial_poses[j](1, 3);
      const auto result_x = ndt_result.pose(0, 3);
      const auto result_y = ndt_result.pose(1, 3);
      ofs_mndt_score << j << "," << nvtl << "," << initial_x << "," << initial_y << "," << result_x << "," << result_y << std::endl;
    }
  }
}
