// Copyright 2022 TIER IV, Inc.
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

/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef PCL_MULTI_VOXEL_GRID_COVARIANCE_OMP_H_
#define PCL_MULTI_VOXEL_GRID_COVARIANCE_OMP_H_

#include <pcl/pcl_macros.h>
#include <pcl/filters/boost.h>
#include <pcl/filters/voxel_grid.h>
#include <map>
#include <unordered_map>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>


#include <future>
#include <ctime>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <pcl/io/pcd_io.h>
#include <queue>
#include <unistd.h>


#ifndef timeDiff
#define timeDiff(start, end)  ((end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec)
#endif

namespace pclomp {
/** \brief A searchable voxel structure containing the mean and covariance of the data.
 * \note For more information please see
 * <b>Magnusson, M. (2009). The Three-Dimensional Normal-Distributions Transform —
 * an Efficient Representation for Registration, Surface Analysis, and Loop Detection.
 * PhD thesis, Orebro University. Orebro Studies in Technology 36</b>
 * \author Brian Okorn (Space and Naval Warfare Systems Center Pacific)
 */
template<typename PointT>
class MultiVoxelGridCovariance : public pcl::VoxelGrid<PointT> {
protected:
  using pcl::VoxelGrid<PointT>::filter_name_;
  using pcl::VoxelGrid<PointT>::getClassName;
  using pcl::VoxelGrid<PointT>::input_;
  using pcl::VoxelGrid<PointT>::indices_;
  using pcl::VoxelGrid<PointT>::filter_limit_negative_;
  using pcl::VoxelGrid<PointT>::filter_limit_min_;
  using pcl::VoxelGrid<PointT>::filter_limit_max_;

  // using pcl::VoxelGrid<PointT>::downsample_all_data_;
  using pcl::VoxelGrid<PointT>::leaf_size_;
  using pcl::VoxelGrid<PointT>::min_b_;
  using pcl::VoxelGrid<PointT>::max_b_;
  using pcl::VoxelGrid<PointT>::inverse_leaf_size_;
  using pcl::VoxelGrid<PointT>::div_b_;
  using pcl::VoxelGrid<PointT>::divb_mul_;

  typedef typename pcl::traits::fieldList<PointT>::type FieldList;
  typedef typename pcl::Filter<PointT>::PointCloud PointCloud;
  typedef typename PointCloud::Ptr PointCloudPtr;
  typedef typename PointCloud::ConstPtr PointCloudConstPtr;

public:
#if PCL_VERSION >= PCL_VERSION_CALC(1, 10, 0)
  typedef pcl::shared_ptr<pcl::VoxelGrid<PointT> > Ptr;
  typedef pcl::shared_ptr<const pcl::VoxelGrid<PointT> > ConstPtr;
#else
  typedef boost::shared_ptr<pcl::VoxelGrid<PointT> > Ptr;
  typedef boost::shared_ptr<const pcl::VoxelGrid<PointT> > ConstPtr;
#endif

  /** \brief Simple structure to hold a centroid, covariance and the number of points in a leaf.
   * Inverse covariance, eigen vectors and eigen values are precomputed. */
  struct Leaf {
    /** \brief Constructor.
     * Sets \ref nr_points_, \ref icov_, \ref mean_ and \ref evals_ to 0 and \ref cov_ and \ref evecs_ to the identity matrix
     */
    Leaf() : nr_points_(0), mean_(Eigen::Vector3d::Zero()), centroid_(), cov_(Eigen::Matrix3d::Identity()), icov_(Eigen::Matrix3d::Zero()), evecs_(Eigen::Matrix3d::Identity()), evals_(Eigen::Vector3d::Zero()) {}

    Leaf(const Leaf& other) : 
      mean_(other.mean_), 
      centroid_(other.centroid_), 
      cov_(other.cov_), 
      icov_(other.icov_), 
      evecs_(other.evecs_),
      evals_(other.evals_)
    {
      nr_points_ = other.nr_points_;
    }

    Leaf(Leaf&& other) :
      mean_(std::move(other.mean_)),
      centroid_(std::move(other.centroid_)), 
      cov_(std::move(other.cov_)), 
      icov_(std::move(other.icov_)), 
      evecs_(std::move(other.evecs_)),
      evals_(std::move(other.evals_))
    {
      nr_points_ = other.nr_points_;
    }

    Leaf& operator=(const Leaf& other)
    {
      mean_ = other.mean_;
      centroid_ = other.centroid_;
      cov_ = other.cov_;
      icov_ = other.icov_;
      evecs_ = other.evecs_;
      evals_ = other.evals_;
      nr_points_ = other.nr_points_;

      return *this;
    }

    Leaf& operator=(Leaf&& other)
    {
      mean_ = std::move(other.mean_);
      centroid_ = std::move(other.centroid_);
      cov_ = std::move(other.cov_);
      icov_ = std::move(other.icov_);
      evecs_ = std::move(other.evecs_);
      evals_ = std::move(other.evals_);
      nr_points_ = other.nr_points_;

      return *this;
    }

    /** \brief Get the voxel covariance.
     * \return covariance matrix
     */
    Eigen::Matrix3d getCov() const {
      return (cov_);
    }

    /** \brief Get the inverse of the voxel covariance.
     * \return inverse covariance matrix
     */
    Eigen::Matrix3d getInverseCov() const {
      return (icov_);
    }

    /** \brief Get the voxel centroid.
     * \return centroid
     */
    Eigen::Vector3d getMean() const {
      return (mean_);
    }

    /** \brief Get the eigen vectors of the voxel covariance.
     * \note Order corresponds with \ref getEvals
     * \return matrix whose columns contain eigen vectors
     */
    Eigen::Matrix3d getEvecs() const {
      return (evecs_);
    }

    /** \brief Get the eigen values of the voxel covariance.
     * \note Order corresponds with \ref getEvecs
     * \return vector of eigen values
     */
    Eigen::Vector3d getEvals() const {
      return (evals_);
    }

    /** \brief Get the number of points contained by this voxel.
     * \return number of points
     */
    int getPointCount() const {
      return (nr_points_);
    }

    /** \brief Number of points contained by voxel */
    int nr_points_;

    /** \brief 3D voxel centroid */
    Eigen::Vector3d mean_;

    /** \brief Nd voxel centroid
     * \note Differs from \ref mean_ when color data is used
     */
    Eigen::VectorXf centroid_;

    /** \brief Voxel covariance matrix */
    Eigen::Matrix3d cov_;

    /** \brief Inverse of voxel covariance matrix */
    Eigen::Matrix3d icov_;

    /** \brief Eigen vectors of voxel covariance matrix */
    Eigen::Matrix3d evecs_;

    /** \brief Eigen values of voxel covariance matrix */
    Eigen::Vector3d evals_;
  };

  /** \brief Pointer to MultiVoxelGridCovariance leaf structure */
  typedef Leaf *LeafPtr;

  /** \brief Const pointer to MultiVoxelGridCovariance leaf structure */
  typedef const Leaf *LeafConstPtr;

  typedef std::map<int64_t, Leaf> LeafDict;

  struct BoundingBox {
    Eigen::Vector4i max;
    Eigen::Vector4i min;
    Eigen::Vector4i div_mul;
  };

  // Each grid contains a vector of leaves and a kdtree built from those leaves
  struct GridNodeType {
    // Tree built on the node's leaves
    pcl::KdTreeFLANN<PointT> kdtree_;
    // Node's leaves
    std::vector<Leaf> leaves_;
    // Center of the node, used later for create the grid kdtree
    PointT center_;
    // Bounding box of the node
    Eigen::Vector3f lower_, upper_;
  };

  using GridNodePtr = std::shared_ptr<GridNodeType>;

public:
  /** \brief Constructor.
   * Sets \ref leaf_size_ to 0
   */
  MultiVoxelGridCovariance() : min_points_per_voxel_(6), min_covar_eigvalue_mult_(0.01) {
    leaf_size_.setZero();
    min_b_.setZero();
    max_b_.setZero();
    filter_name_ = "MultiVoxelGridCovariance";

    setThreadNum(1);
    last_check_tid_ = -1;
    grid_radius_ = 0;
    rebuilt_ = false;
  }

  MultiVoxelGridCovariance(const MultiVoxelGridCovariance &other);
  MultiVoxelGridCovariance(MultiVoxelGridCovariance &&other);

  MultiVoxelGridCovariance &operator=(const MultiVoxelGridCovariance &other);
  MultiVoxelGridCovariance &operator=(MultiVoxelGridCovariance &&other);

  /** \brief Add a cloud to the voxel grid list and build a ND voxel grid from it.
   */
  void setInputCloudAndFilter(const PointCloudConstPtr &cloud, const std::string &grid_id);

  /** \brief Remove a ND voxel grid corresponding to the specified id
   */
  void removeCloud(const std::string &grid_id);

  /** \brief Build Kdtrees from the NDT voxel for later radius search
   */
  void createKdtree();

  /** \brief Search for all the nearest occupied voxels of the query point in a given radius.
   * \note Only voxels containing a sufficient number of points are used.
   * \param[in] point the given query point
   * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
   * \param[out] k_leaves the resultant leaves of the neighboring points
   * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
   * \param[in] max_nn
   * \return number of neighbors found
   */
  int radiusSearch(const PointT &point, double radius, std::vector<LeafConstPtr> &k_leaves, unsigned int max_nn = 0) const;

  /** \brief Search for all the nearest occupied voxels of the query point in a given radius.
   * \note Only voxels containing a sufficient number of points are used.
   * \param[in] cloud the given query point
   * \param[in] index a valid index in cloud representing a valid (i.e., finite) query point
   * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
   * \param[out] k_leaves the resultant leaves of the neighboring points
   * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
   * \param[in] max_nn
   * \return number of neighbors found
   */
  inline int radiusSearch(const PointCloud &cloud, int index, double radius, std::vector<LeafConstPtr> &k_leaves, unsigned int max_nn = 0) const {
    if(index >= static_cast<int>(cloud.size()) || index < 0) return (0);
    return (radiusSearch(cloud[index], radius, k_leaves, max_nn));
  }

  PointCloud getVoxelPCD() {
    if (!rebuilt_)
    {
      // First, count the number of leaves
      int64_t total_leaf_num = 0;

      for (auto& grid : grid_list_)
      {
        total_leaf_num += grid->leaves_.size();
      }

      // Now generate the voxel centroid cloud for a kdtree
      voxel_centroids_ptr_.reset(new PointCloud);
      voxel_centroids_ptr_->height = 1;
      voxel_centroids_ptr_->is_dense = true;
      voxel_centroids_ptr_->resize(total_leaf_num);

      size_t pos = 0;

      for (auto& grid : grid_list_)
      {
        for (auto& leaf : grid->leaves_)
        {
          auto& op = (*voxel_centroids_ptr_)[pos++];

          op.x = leaf.centroid_[0];
          op.y = leaf.centroid_[1];
          op.z = leaf.centroid_[2];
        }
      }

      voxel_centroids_ptr_->width = static_cast<uint32_t>(total_leaf_num);
      // Next time if no cloud change, no need to re-do steps above
      rebuilt_ = true;
    }

    return *voxel_centroids_ptr_;
  }

  std::vector<std::string> getCurrentMapIDs() const {
    std::vector<std::string> output;

    for(const auto &element : sid_to_iid_) {
      output.push_back(element.first);
    }

    return output;
  }

  void setThreadNum(int thread_num)
  {
    if (thread_num <= 0)
    {
      thread_num_ = 1;
    }

    thread_num_ = thread_num;
    thread_futs_.resize(thread_num_);
    processing_inputs_.resize(thread_num_);
  }

  ~MultiVoxelGridCovariance()
  {
    // Stop all threads in the case an intermediate interrupt occurs
    sync();
  }

protected:
  // Return the index of an idle thread, which is not running any
  // job, or has already finished its job and waiting for a join.
  // In the later case, join the thread and
  int get_idle_tid()
  {
    int tid = (last_check_tid_ == thread_num_ - 1) ? 0 : last_check_tid_ + 1;
    std::chrono::microseconds span(50);

    // Loop until an idle thread is found
    while (true)
    {
      // gettimeofday(&start, NULL);
      // Return immediately if a thread that has not been given a job is found
      if (!thread_futs_[tid].valid())
      {
        last_check_tid_ = tid;
        return tid;
      }

      // If no such thread is found, wait for the current thread to finish its job
      if (thread_futs_[tid].wait_for(span) == std::future_status::ready)
      {
        last_check_tid_ = tid;
        return tid;
      }

      // If the current thread has not finished its job, check the next thread
      tid = (tid == thread_num_ - 1) ? 0 : tid + 1;
    }
  }

  // Wait for all running threads to finish
  void sync()
  {
    for (int i = 0; i < thread_num_; ++i)
    {
      if (thread_futs_[i].valid())
      {
        thread_futs_[i].wait();
      }
    }
  }

  // Compute the square distance from a point to a bounding box 
  inline double sqr_dist_to_grid(const PointT& p, const GridNodeType& grid) const
  {
    double dx = (p.x < grid.lower_[0]) ? grid.lower_[0] - p.x : ((p.x > grid.upper_[0]) ? p.x - grid.upper_[0] : 0); 
    double dy = (p.y < grid.lower_[1]) ? grid.lower_[1] - p.y : ((p.y > grid.upper_[1]) ? p.y - grid.upper_[1] : 0); 
    double dz = (p.z < grid.lower_[2]) ? grid.lower_[2] - p.z : ((p.z > grid.upper_[2]) ? p.z - grid.upper_[2] : 0); 

    return (dx * dx + dy * dy + dz * dz);
  }

  // A wraper of the real applyFilter
  bool applyFilterThread(int tid, GridNodeType &node) 
  {
    applyFilter(processing_inputs_[tid], node);
    processing_inputs_[tid].reset();

    return true;
  }

  /** \brief Filter cloud and initializes voxel structure.
   * \param[out] output cloud containing centroids of voxels containing a sufficient number of points
   */
  void applyFilter(const PointCloudConstPtr &input, GridNodeType &node);

  void updateLeaf(const PointT &point, const int &centroid_size, Leaf &leaf) const;

  void computeLeafParams(const Eigen::Vector3d &pt_sum, Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> &eigensolver, Leaf &leaf) const;

  int64_t getLeafID(const PointT &point, const BoundingBox &bbox) const;

  /** \brief Minimum points contained with in a voxel to allow it to be usable. */
  int min_points_per_voxel_;

  /** \brief Minimum allowable ratio between eigenvalues to prevent singular covariance matrices. */
  double min_covar_eigvalue_mult_;

  // Indicate whether the voxel centroid cloud was rebuilt or not
  // Whenever a cloud is added or remove, this is switched to false
  // Whenever a getVoxelPCD is called, this is switched to true
  bool rebuilt_;
  PointCloudPtr voxel_centroids_ptr_;

  // Thread pooling, for parallel processing
  int thread_num_;
  std::vector<std::future<bool>> thread_futs_;
  std::vector<PointCloudConstPtr> processing_inputs_;
  int last_check_tid_;

  // TODO: no need a map to hold the grid_list_
  // A vector is a much better option to avoid log(n) access
  // A map to convert string index to integer index, used for grids
  std::map<std::string, int> sid_to_iid_;
  // Grids of leaves are held in a vector for faster access speed
  std::vector<GridNodePtr> grid_list_;
  // A 2-step access is used to go for a leaf, given its 1D index
  // First, binary search the grid_offset_ to find the index of the grid containing the leaf
  // Second, go to grid_list_ at the found index, and access the leaf
  // No need several deep copies as well as map access
  // A cloud made by centers of grids
  PointCloudPtr grid_centers_ptr_;
  // A kdtree built from the centers of grids
  pcl::KdTreeFLANN<PointT> grid_kdtree_;
  // The radius used for searching candidate grids
  // It is the maximum half length of the diagonals of grids' bounding boxes
  float grid_radius_; 

};
}  // namespace pclomp

#endif  // #ifndef PCL_MULTI_VOXEL_GRID_COVARIANCE_H_
