#include "estimate_covariance/estimate_covariance.hpp"

namespace pclomp {

Eigen::Matrix2d find_rotation_matrix_aligning_covariance_to_principal_axes(const Eigen::Matrix2d& matrix) {
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(matrix);
  if(eigensolver.info() == Eigen::Success) {
    const Eigen::Vector2d eigen_vec = eigensolver.eigenvectors().col(0);
    const double th = std::atan2(eigen_vec.y(), eigen_vec.x());
    return Eigen::Rotation2Dd(th).toRotationMatrix();
  }
  throw std::runtime_error("Eigen solver failed. Return output_pose_covariance value.");
}

Eigen::Matrix2d estimate_xy_covariance_by_Laplace_approximation(const NdtResult& ndt_result) {
  const Eigen::Matrix<double, 6, 6>& hessian = ndt_result.hessian;
  const Eigen::Matrix2d hessian_xy = hessian.block<2, 2>(0, 0);
  const Eigen::Matrix2d covariance_xy = -hessian_xy.inverse();
  return covariance_xy;
}

Eigen::Matrix2d estimate_xy_covariance_by_multi_ndt(const NdtResult& ndt_result, std::shared_ptr<pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>> ndt_ptr, const Eigen::Matrix4f& initial_pose, const std::vector<double>& offset_x, const std::vector<double>& offset_y) {
  const Eigen::Matrix2d cov_by_la = estimate_xy_covariance_by_Laplace_approximation(ndt_result);
  const Eigen::Matrix2d rot = find_rotation_matrix_aligning_covariance_to_principal_axes(-cov_by_la);

  assert(offset_x.size() == offset_y.size());

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
  for(int i = 0; i < n - 1; i++) {
    const Eigen::Vector2d pose_offset(offset_x[i], offset_y[i]);
    const Eigen::Vector2d rotated_pose_offset_2d = rot * pose_offset;

    Eigen::Matrix4f sub_initial_pose_matrix(Eigen::Matrix4f::Identity());
    sub_initial_pose_matrix = ndt_result.pose;
    sub_initial_pose_matrix(0, 3) += static_cast<float>(rotated_pose_offset_2d.x());
    sub_initial_pose_matrix(1, 3) += static_cast<float>(rotated_pose_offset_2d.y());

    auto sub_output_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    ndt_ptr->align(*sub_output_cloud, sub_initial_pose_matrix);
    const Eigen::Matrix4f sub_ndt_result = ndt_ptr->getResult().pose;

    const Eigen::Vector2d sub_ndt_pose_2d = sub_ndt_result.topRightCorner<2, 1>().cast<double>();
    mean += sub_ndt_pose_2d;
    ndt_pose_2d_vec.emplace_back(sub_ndt_pose_2d);

    initial_pose_vec.push_back(sub_initial_pose_matrix);
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
  return pca_covariance;
}

}  // namespace pclomp
