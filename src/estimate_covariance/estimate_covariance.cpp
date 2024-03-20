#include "estimate_covariance/estimate_covariance.hpp"
#include <fstream>
#include <iomanip>

namespace pclomp {

Eigen::Matrix2d estimate_xy_covariance_by_Laplace_approximation(const Eigen::Matrix<double, 6, 6>& hessian) {
  const Eigen::Matrix2d hessian_xy = hessian.block<2, 2>(0, 0);
  const Eigen::Matrix2d covariance_xy = -hessian_xy.inverse();
  return covariance_xy;
}

ResultOfMultiNdtCovarianceEstimation estimate_xy_covariance_by_multi_ndt(const NdtResult& ndt_result, std::shared_ptr<pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>> ndt_ptr, const std::vector<Eigen::Matrix4f>& poses_to_search) {
  // initialize by the main result
  const Eigen::Vector2d ndt_pose_2d(ndt_result.pose(0, 3), ndt_result.pose(1, 3));
  std::vector<Eigen::Vector2d> ndt_pose_2d_vec{ndt_pose_2d};

  // multiple searches
  std::vector<NdtResult> ndt_results;
  for(const Eigen::Matrix4f& curr_pose : poses_to_search) {
    auto sub_output_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    ndt_ptr->align(*sub_output_cloud, curr_pose);
    ndt_results.push_back(ndt_ptr->getResult());
    const Eigen::Matrix4f sub_ndt_result = ndt_ptr->getResult().pose;
    const Eigen::Vector2d sub_ndt_pose_2d = sub_ndt_result.topRightCorner<2, 1>().cast<double>();
    ndt_pose_2d_vec.emplace_back(sub_ndt_pose_2d);
  }

  // calculate the weights
  const int n = static_cast<int>(ndt_results.size()) + 1;
  const std::vector<double> weight_vec(n, 1.0 / n);

  // calculate mean and covariance
  auto [mean, covariance] = calculate_weighted_mean_and_cov(ndt_pose_2d_vec, weight_vec);

  // unbiased covariance
  covariance *= static_cast<double>(n - 1) / n;

  return {mean, covariance, poses_to_search, ndt_results};
}

ResultOfMultiNdtCovarianceEstimation estimate_xy_covariance_by_multi_ndt_score(const NdtResult& ndt_result, std::shared_ptr<pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>> ndt_ptr, const std::vector<Eigen::Matrix4f>& poses_to_search) {
  // initialize by the main result
  const Eigen::Vector2d ndt_pose_2d(ndt_result.pose(0, 3), ndt_result.pose(1, 3));
  std::vector<Eigen::Vector2d> ndt_pose_2d_vec{ndt_pose_2d};
  std::vector<double> score_vec{ndt_result.nearest_voxel_transformation_likelihood};

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
    const Eigen::Vector2d sub_ndt_pose_2d(curr_pose(0, 3), curr_pose(1, 3));
    ndt_pose_2d_vec.emplace_back(sub_ndt_pose_2d);
    score_vec.emplace_back(sub_ndt_result.nearest_voxel_transformation_likelihood);
  }

  // set itr to original
  ndt_ptr->setMaximumIterations(original_max_itr);

  // calculate the weights
  const std::vector<double> weight_vec = calc_weight_vec(score_vec, 0.1);

  // calculate mean and covariance
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

std::vector<Eigen::Matrix4f> propose_poses_to_search(const NdtResult& ndt_result, const std::vector<double>& offset_x, const std::vector<double>& offset_y) {
  assert(offset_x.size() == offset_y.size());
  const Eigen::Matrix<double, 6, 6>& hessian = ndt_result.hessian;
  const Eigen::Matrix4f& center_pose = ndt_result.pose;
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

}  // namespace pclomp
