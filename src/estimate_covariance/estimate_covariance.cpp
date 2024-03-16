#include "estimate_covariance/estimate_covariance.hpp"
#include <fstream>
#include <iomanip>

namespace pclomp {

Eigen::Matrix2d estimate_xy_covariance_by_Laplace_approximation(const Eigen::Matrix<double, 6, 6>& hessian) {
  const Eigen::Matrix2d hessian_xy = hessian.block<2, 2>(0, 0);
  const Eigen::Matrix2d covariance_xy = -hessian_xy.inverse();
  return covariance_xy;
}

ResultOfMultiNdtCovarianceEstimation estimate_xy_covariance_by_multi_ndt(const NdtResult& ndt_result, std::shared_ptr<pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>> ndt_ptr, const Eigen::Matrix4f& initial_pose, const std::vector<double>& offset_x, const std::vector<double>& offset_y) {
  const std::vector<Eigen::Matrix4f> poses_to_search = propose_poses_to_search(ndt_result.hessian, ndt_result.pose, offset_x, offset_y);

  // first result is added to mean
  const int n = static_cast<int>(offset_x.size()) + 1;
  const Eigen::Vector2d ndt_pose_2d(ndt_result.pose(0, 3), ndt_result.pose(1, 3));
  Eigen::Vector2d mean = ndt_pose_2d;
  std::vector<Eigen::Vector2d> ndt_pose_2d_vec;
  ndt_pose_2d_vec.reserve(n);
  ndt_pose_2d_vec.emplace_back(ndt_pose_2d);

  std::vector<Eigen::Matrix4f> initial_pose_vec = {initial_pose};
  std::vector<Eigen::Matrix4f> ndt_result_pose_vec = {ndt_result.pose};

  // multiple searches
  std::vector<NdtResult> ndt_results;
  for(const Eigen::Matrix4f& curr_pose : poses_to_search) {
    auto sub_output_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    ndt_ptr->align(*sub_output_cloud, curr_pose);
    ndt_results.push_back(ndt_ptr->getResult());
    const Eigen::Matrix4f sub_ndt_result = ndt_ptr->getResult().pose;
    const Eigen::Vector2d sub_ndt_pose_2d = sub_ndt_result.topRightCorner<2, 1>().cast<double>();
    mean += sub_ndt_pose_2d;
    ndt_pose_2d_vec.emplace_back(sub_ndt_pose_2d);
    initial_pose_vec.push_back(curr_pose);
    ndt_result_pose_vec.push_back(sub_ndt_result);
  }

  // calculate the covariance matrix
  mean /= n;
  Eigen::Matrix2d pca_covariance = Eigen::Matrix2d::Zero();
  for(const auto& temp_ndt_pose_2d : ndt_pose_2d_vec) {
    const Eigen::Vector2d diff_2d = temp_ndt_pose_2d - mean;
    pca_covariance += diff_2d * diff_2d.transpose();
  }
  pca_covariance /= (n - 1);  // unbiased covariance
  return {mean, pca_covariance, poses_to_search, ndt_results};
}

ResultOfMultiNdtCovarianceEstimation estimate_xy_covariance_by_multi_ndt_score(const NdtResult& ndt_result, std::shared_ptr<pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>> ndt_ptr, const Eigen::Matrix4f& initial_pose, const std::vector<double>& offset_x, const std::vector<double>& offset_y) {
  const std::vector<Eigen::Matrix4f> poses_to_search = propose_poses_to_search(ndt_result.hessian, ndt_result.pose, offset_x, offset_y);

  std::vector<Eigen::Vector2d> ndt_pose_2d_vec;
  std::vector<double> score_vec;

  const int primary_ndt_itr = ndt_result.transformation_array.size();
  for(int i = 0; i < primary_ndt_itr; i++) {
    const Eigen::Vector2d ndt_pose_2d(ndt_result.transformation_array[i](0, 3), ndt_result.transformation_array[i](1, 3));
    ndt_pose_2d_vec.emplace_back(ndt_pose_2d);
    score_vec.emplace_back(ndt_result.nearest_voxel_transformation_likelihood_array[i]);
  }

  // set itr to 1
  const int original_max_itr = ndt_ptr->getMaximumIterations();
  ndt_ptr->setMaximumIterations(1);

  // multiple searches
  std::vector<NdtResult> ndt_results;
  for(const Eigen::Matrix4f& curr_pose : poses_to_search) {
    auto sub_output_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    ndt_ptr->align(*sub_output_cloud, curr_pose);
    const NdtResult sub_ndt_result = ndt_ptr->getResult();
    ndt_results.push_back(sub_ndt_result);
    const Eigen::Matrix4f sub_ndt_pose = sub_ndt_result.transformation_array[0];
    const Eigen::Vector2d sub_ndt_pose_2d(sub_ndt_pose(0, 3), sub_ndt_pose(1, 3));
    ndt_pose_2d_vec.emplace_back(sub_ndt_pose_2d);
    score_vec.emplace_back(sub_ndt_result.nearest_voxel_transformation_likelihood_array[0]);
  }

  // set itr to original
  ndt_ptr->setMaximumIterations(original_max_itr);

  // calculate the weights
  const std::vector<double> weight_vec = calc_weight_vec(score_vec, 0.1);

  // calculate the covariance matrix
  const auto [mean, covariance] = calculate_weighted_mean_and_cov(ndt_pose_2d_vec, weight_vec);
  return {mean, covariance, poses_to_search, ndt_results};
}

Eigen::Matrix2d find_rotation_matrix_aligning_covariance_to_principal_axes(const Eigen::Matrix2d& matrix) {
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(matrix);
  if(eigensolver.info() == Eigen::Success) {
    const Eigen::Vector2d eigen_vec = eigensolver.eigenvectors().col(0);
    const double th = std::atan2(eigen_vec.y(), eigen_vec.x());
    return Eigen::Rotation2Dd(th).toRotationMatrix();
  }
  throw std::runtime_error("Eigen solver failed. Return output_pose_covariance value.");
}

std::vector<Eigen::Matrix4f> propose_poses_to_search(const Eigen::Matrix<double, 6, 6>& hessian, const Eigen::Matrix4f& center_pose, const std::vector<double>& offset_x, const std::vector<double>& offset_y) {
  assert(offset_x.size() == offset_y.size());
  const Eigen::Matrix2d covariance = estimate_xy_covariance_by_Laplace_approximation(hessian);
  const Eigen::Matrix2d rot = find_rotation_matrix_aligning_covariance_to_principal_axes(-covariance);
  std::vector<Eigen::Matrix4f> poses_to_search;
  for(int i = 0; i < static_cast<int>(offset_x.size()); i++) {
    const Eigen::Vector2d pose_offset(offset_x[i], offset_y[i]);
    const Eigen::Vector2d rotated_pose_offset_2d = rot * pose_offset;
    Eigen::Matrix4f curr_pose = center_pose;
    curr_pose(0, 3) += static_cast<float>(rotated_pose_offset_2d.x());
    curr_pose(1, 3) += static_cast<float>(rotated_pose_offset_2d.y());
    poses_to_search.emplace_back(curr_pose);
  }
  return poses_to_search;
}

std::vector<double> calc_weight_vec(const std::vector<double>& score_vec, double temperature) {
  const int n = static_cast<int>(score_vec.size());
  const double max_score = *std::max_element(score_vec.begin(), score_vec.end());
  std::vector<double> exp_score_vec(n);
  double exp_score_sum = 0.0;
  for(int i = 0; i < n; i++) {
    exp_score_vec[i] = std::exp((score_vec[i] - max_score) / temperature);
    exp_score_sum += exp_score_vec[i];
  }
  for(int i = 0; i < n; i++) {
    exp_score_vec[i] /= exp_score_sum;
  }
  return exp_score_vec;
}

std::pair<Eigen::Vector2d, Eigen::Matrix2d> calculate_weighted_mean_and_cov(const std::vector<Eigen::Vector2d>& pose_2d_vec, const std::vector<double>& weight_vec) {
  const int n = static_cast<int>(pose_2d_vec.size());
  Eigen::Vector2d mean = Eigen::Vector2d::Zero();
  for(int i = 0; i < n; i++) {
    mean += weight_vec[i] * pose_2d_vec[i];
  }
  Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();
  for(int i = 0; i < n; i++) {
    const Eigen::Vector2d diff = pose_2d_vec[i] - mean;
    covariance += weight_vec[i] * diff * diff.transpose();
  }
  return {mean, covariance};
}

void output_pose_score_weight(const std::vector<Eigen::Vector2d>& pose_2d_vec, const std::vector<double>& score_vec, const std::vector<double>& weight_vec) {
  static int counter = 0;
  std::stringstream ss;
  ss << "log" << std::setw(8) << std::setfill('0') << counter++ << ".csv";
  std::ofstream ofs(ss.str());
  ofs << std::fixed;
  ofs << "x,y,score,weight" << std::endl;  // header
  const int n = static_cast<int>(pose_2d_vec.size());
  for(int i = 0; i < n; i++) {
    ofs << pose_2d_vec[i].x() << "," << pose_2d_vec[i].y() << "," << score_vec[i] << "," << weight_vec[i] << std::endl;
  }
}

}  // namespace pclomp
