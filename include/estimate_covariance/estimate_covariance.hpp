#ifndef NDT_OMP__ESTIMATE_COVARIANCE_HPP_
#define NDT_OMP__ESTIMATE_COVARIANCE_HPP_

#include <Eigen/Core>
#include "multigrid_pclomp/multigrid_ndt_omp.h"

namespace pclomp {

Eigen::Matrix2d find_rotation_matrix_aligning_covariance_to_principal_axes(const Eigen::Matrix2d& matrix);
Eigen::Matrix2d estimate_xy_covariance_by_Laplace_approximation(const NdtResult& ndt_result);
Eigen::Matrix2d estimate_xy_covariance_by_multi_ndt(const NdtResult& ndt_result, std::shared_ptr<pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>> ndt_ptr, const Eigen::Matrix4f& initial_pose, const std::vector<double>& offset_x, const std::vector<double>& offset_y);
Eigen::Matrix2d estimate_xy_covariance_by_multi_ndt_score(const NdtResult& ndt_result, std::shared_ptr<pclomp::MultiGridNormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>> ndt_ptr, const Eigen::Matrix4f& initial_pose, const std::vector<double>& offset_x, const std::vector<double>& offset_y);

/** \brief Calculate weighted mean and covariance */
std::pair<Eigen::Vector2d, Eigen::Matrix2d> calculate_weighted_mean_and_cov(const std::vector<Eigen::Vector2d>& pose_2d_vec, const std::vector<double>& score_vec);

}  // namespace pclomp

#endif  // NDT_OMP__ESTIMATE_COVARIANCE_HPP_
