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
  std::shared_ptr<pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>> mg_ndt_omp(new pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>());
  mg_ndt_omp->setInputTarget(target_cloud);
  mg_ndt_omp->setResolution(2.0);
  mg_ndt_omp->setNumThreads(4);
  mg_ndt_omp->setMaximumIterations(30);
  mg_ndt_omp->setTransformationEpsilon(0.01);
  mg_ndt_omp->setStepSize(0.1);
  mg_ndt_omp->createVoxelKdtree();

  std::cout << std::fixed;

  const std::vector<double> offset_x = {0.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.0, 2.0, -2.0};
  const std::vector<double> offset_y = {0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0};

  // output result
  std::filesystem::create_directories(output_dir);
  std::ofstream ofs(output_dir + "/result.csv");
  ofs << std::fixed;
  ofs << "index,score,";
  ofs << "initial_x,initial_y,initial_yaw,";
  ofs << "result_x,result_y,result_yaw,";
  ofs << "elapsed_la,cov_by_la_00,cov_by_la_01,cov_by_la_10,cov_by_la_11,";
  ofs << "elapsed_mndt,cov_by_mndt_00,cov_by_mndt_01,cov_by_mndt_10,cov_by_mndt_11,";
  ofs << "elapsed_mndt_score,cov_by_mndt_score_00,cov_by_mndt_score_01,cov_by_mndt_score_10,cov_by_mndt_score_11,";
  ofs << "cov_by_la_rotated_00,cov_by_la_rotated_01,cov_by_la_rotated_10,cov_by_la_rotated_11,";
  ofs << "cov_by_mndt_rotated_00,cov_by_mndt_rotated_01,cov_by_mndt_rotated_10,cov_by_mndt_rotated_11,";
  ofs << "cov_by_mndt_score_rotated_00,cov_by_mndt_score_rotated_01,cov_by_mndt_score_rotated_10,cov_by_mndt_score_rotated_11" << std::endl;

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

    const std::vector<Eigen::Matrix4f> poses_to_search = pclomp::propose_poses_to_search(ndt_result, offset_x, offset_y);

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
    const double temperature = 0.1;
    t1 = std::chrono::system_clock::now();
    const pclomp::ResultOfMultiNdtCovarianceEstimation result_of_mndt_score = pclomp::estimate_xy_covariance_by_multi_ndt_score(ndt_result, mg_ndt_omp, poses_to_search, temperature);
    const Eigen::Matrix2d cov_by_mndt_score = result_of_mndt_score.covariance;
    t2 = std::chrono::system_clock::now();
    const auto elapsed_mndt_score = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;

    // output result
    const auto result_x = ndt_result.pose(0, 3);
    const auto result_y = ndt_result.pose(1, 3);
    const Eigen::Vector3f euler_initial = initial_pose.block<3, 3>(0, 0).eulerAngles(0, 1, 2);
    const Eigen::Vector3f euler_result = ndt_result.pose.block<3, 3>(0, 0).eulerAngles(0, 1, 2);
    const Eigen::Matrix2d cov_by_la_rotated = pclomp::rotate_covariance_to_base_link(cov_by_la, ndt_result.pose);
    const Eigen::Matrix2d cov_by_mndt_rotated = pclomp::rotate_covariance_to_base_link(cov_by_mndt, ndt_result.pose);
    const Eigen::Matrix2d cov_by_mndt_score_rotated = pclomp::rotate_covariance_to_base_link(cov_by_mndt_score, ndt_result.pose);
    ofs << i << "," << score << ",";
    ofs << initial_pose(0, 3) << "," << initial_pose(1, 3) << "," << euler_initial(2) << ",";
    ofs << result_x << "," << result_y << "," << euler_result(2) << ",";
    ofs << elapsed_la << "," << cov_by_la(0, 0) << "," << cov_by_la(0, 1) << "," << cov_by_la(1, 0) << "," << cov_by_la(1, 1) << ",";
    ofs << elapsed_mndt << "," << cov_by_mndt(0, 0) << "," << cov_by_mndt(0, 1) << "," << cov_by_mndt(1, 0) << "," << cov_by_mndt(1, 1) << ",";
    ofs << elapsed_mndt_score << "," << cov_by_mndt_score(0, 0) << "," << cov_by_mndt_score(0, 1) << "," << cov_by_mndt_score(1, 0) << "," << cov_by_mndt_score(1, 1) << ",";
    ofs << cov_by_la_rotated(0, 0) << "," << cov_by_la_rotated(0, 1) << "," << cov_by_la_rotated(1, 0) << "," << cov_by_la_rotated(1, 1) << ",";
    ofs << cov_by_mndt_rotated(0, 0) << "," << cov_by_mndt_rotated(0, 1) << "," << cov_by_mndt_rotated(1, 0) << "," << cov_by_mndt_rotated(1, 1) << ",";
    ofs << cov_by_mndt_score_rotated(0, 0) << "," << cov_by_mndt_score_rotated(0, 1) << "," << cov_by_mndt_score_rotated(1, 0) << "," << cov_by_mndt_score_rotated(1, 1) << std::endl;

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
