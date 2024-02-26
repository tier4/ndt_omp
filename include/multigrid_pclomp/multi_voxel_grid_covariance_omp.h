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
     * Sets \ref nr_points, \ref icov_, \ref mean_ and \ref evals_ to 0 and \ref cov_ and \ref evecs_ to the identity matrix
     */
    Leaf() : nr_points(0), mean_(Eigen::Vector3d::Zero()), centroid(), cov_(Eigen::Matrix3d::Identity()), icov_(Eigen::Matrix3d::Zero()), evecs_(Eigen::Matrix3d::Identity()), evals_(Eigen::Vector3d::Zero()) {}

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
      return (nr_points);
    }

    /** \brief Number of points contained by voxel */
    int nr_points;

    /** \brief 3D voxel centroid */
    Eigen::Vector3d mean_;

    /** \brief Nd voxel centroid
     * \note Differs from \ref mean_ when color data is used
     */
    Eigen::VectorXf centroid;

    /** \brief Voxel covariance matrix */
    Eigen::Matrix3d cov_;

    /** \brief Inverse of voxel covariance matrix */
    Eigen::Matrix3d icov_;

    /** \brief Eigen vectors of voxel covariance matrix */
    Eigen::Matrix3d evecs_;

    /** \brief Eigen values of voxel covariance matrix */
    Eigen::Vector3d evals_;
  };

  struct LeafID {
    std::string parent_grid_id;
    int leaf_index;
    bool operator<(const LeafID &rhs) const {
      if(parent_grid_id < rhs.parent_grid_id) {
        return true;
      }
      if(parent_grid_id > rhs.parent_grid_id) {
        return false;
      }
      if(leaf_index < rhs.leaf_index) {
        return true;
      }
      if(leaf_index > rhs.leaf_index) {
        return false;
      }
      return false;
    }
  };

  /** \brief Pointer to MultiVoxelGridCovariance leaf structure */
  typedef Leaf *LeafPtr;

  /** \brief Const pointer to MultiVoxelGridCovariance leaf structure */
  typedef const Leaf *LeafConstPtr;

  typedef std::map<LeafID, Leaf> LeafDict;

  struct BoundingBox {
    Eigen::Vector4i max;
    Eigen::Vector4i min;
    Eigen::Vector4i div_mul;
  };

public:
  /** \brief Constructor.
   * Sets \ref leaf_size_ to 0
   */
  MultiVoxelGridCovariance() : min_points_per_voxel_(6), min_covar_eigvalue_mult_(0.01), leaves_(), grid_leaves_(), leaf_indices_(), kdtree_() {
    leaf_size_.setZero();
    min_b_.setZero();
    max_b_.setZero();
    filter_name_ = "MultiVoxelGridCovariance";

    thread_num_ = 1;
    setThreadNum(thread_num_);
    last_check_tid_ = -1;

    gettimeofday(&all_start_, NULL);

    test_file_.open("/home/anh/Work/autoware/time_test.txt", std::ios::app);
  }

  MultiVoxelGridCovariance(const MultiVoxelGridCovariance &other);
  MultiVoxelGridCovariance(MultiVoxelGridCovariance &&other);

  MultiVoxelGridCovariance &operator=(const MultiVoxelGridCovariance &other);
  MultiVoxelGridCovariance &operator=(MultiVoxelGridCovariance &&other);

  /** \brief Initializes voxel structure.
   */
  inline void setInputCloudAndFilter(const PointCloudConstPtr &cloud, const std::string &grid_id) {
    LeafDict leaves;
    applyFilter(cloud, grid_id, leaves);

    grid_leaves_[grid_id] = leaves;
  }

  inline void setInputCloudAndFilter2(const PointCloudConstPtr& cloud, const std::string &grid_id) {    
    int idle_tid = get_idle_tid();
    processing_inputs_[idle_tid] = cloud;
    thread_futs_[idle_tid] = std::async(std::launch::async, 
                                          &MultiVoxelGridCovariance<PointT>::applyFilterThread, this, 
                                          idle_tid, std::cref(grid_id), std::ref(grid_leaves_[grid_id]));
  }

  inline void removeCloud(const std::string &grid_id) {
    grid_leaves_.erase(grid_id);
  }

  inline void createKdtree() {
    // Wait for all threads to finish
    sync();

    gettimeofday(&all_end_, NULL);

    // No need mutex here, since no other thread is running now
    test_file_ << "Total filter time = " << timeDiff(all_start_, all_end_) << std::endl;

    // Measure time of building tree
    gettimeofday(&all_start_, NULL);
    leaves_.clear();
    for(const auto &kv : grid_leaves_) {
      test_file_ << "Grid leaves size = " << kv.second.size() << std::endl;
      leaves_.insert(kv.second.begin(), kv.second.end());
    }

    leaf_indices_.clear();
    voxel_centroids_ptr_.reset(new PointCloud);
    voxel_centroids_ptr_->height = 1;
    voxel_centroids_ptr_->is_dense = true;
    voxel_centroids_ptr_->points.clear();
    voxel_centroids_ptr_->points.reserve(leaves_.size());
    for(const auto &element : leaves_) {
      leaf_indices_.push_back(element.first);
      voxel_centroids_ptr_->push_back(PointT());
      voxel_centroids_ptr_->points.back().x = element.second.centroid[0];
      voxel_centroids_ptr_->points.back().y = element.second.centroid[1];
      voxel_centroids_ptr_->points.back().z = element.second.centroid[2];
    }
    voxel_centroids_ptr_->width = static_cast<uint32_t>(voxel_centroids_ptr_->points.size());

    if(voxel_centroids_ptr_->size() > 0) {
      kdtree_.setInputCloud(voxel_centroids_ptr_);
    }
    gettimeofday(&all_end_, NULL);

    test_file_ << "Build tree time = " << timeDiff(all_start_, all_end_) << std::endl;

    if (!voxel_centroids_ptr_)
    {
      test_file_ << "Empty centroids" << std::endl;
    }
    else
    {
      test_file_ << "Number of centroids = " << voxel_centroids_ptr_->size();
    }

    gettimeofday(&all_start_, NULL);

    // For debug, save the centroid cloud
    pcl::io::savePCDFileBinary("/home/anh/Work/autoware/centroids.pcd", *voxel_centroids_ptr_);

    // Save all grids
    for (auto &gl : grid_leaves_)
    {
      leafToPCD(gl.first, gl.second);
    }
    exit(0);
    // End
  }

  /** \brief Search for all the nearest occupied voxels of the query point in a given radius.
   * \note Only voxels containing a sufficient number of points are used.
   * \param[in] point the given query point
   * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
   * \param[out] k_leaves the resultant leaves of the neighboring points
   * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
   * \param[in] max_nn
   * \return number of neighbors found
   */
  int radiusSearch(const PointT &point, double radius, std::vector<LeafConstPtr> &k_leaves, std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const {
    k_leaves.clear();

    // Find neighbors within radius in the occupied voxel centroid cloud
    std::vector<int> k_indices;
    int k = kdtree_.radiusSearch(point, radius, k_indices, k_sqr_distances, max_nn);

    // Find leaves corresponding to neighbors
    k_leaves.reserve(k);
    for(std::vector<int>::iterator iter = k_indices.begin(); iter != k_indices.end(); iter++) {
      auto leaf = leaves_.find(leaf_indices_[*iter]);
      if(leaf == leaves_.end()) {
        std::cerr << "error : could not find the leaf corresponding to the voxel" << std::endl;
        std::cin.ignore(1);
      }
      k_leaves.push_back(&(leaf->second));
    }
    return k;
  }

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
  inline int radiusSearch(const PointCloud &cloud, int index, double radius, std::vector<LeafConstPtr> &k_leaves, std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const {
    if(index >= static_cast<int>(cloud.points.size()) || index < 0) return (0);
    return (radiusSearch(cloud.points[index], radius, k_leaves, k_sqr_distances, max_nn));
  }

  PointCloud getVoxelPCD() const {
    return *voxel_centroids_ptr_;
  }

  std::vector<std::string> getCurrentMapIDs() const {
    std::vector<std::string> output{};
    for(const auto &element : grid_leaves_) {
      output.push_back(element.first);
    }
    return output;
  }

  void setThreadNum(int thread_num)
  {
    test_file_ << __FILE__ << "::" << __LINE__ << "::" << __func__ << "::thread num = " << thread_num << std::endl;
    if (thread_num <= 0)
    {
      thread_num_ = 1;
    }

    thread_num_ = thread_num;
    thread_futs_.resize(thread_num_);
    processing_inputs_.resize(thread_num_);
  }

protected:
  // Return the index of an idle thread, which is not running any
  // job, or has already finished its job and waiting for a join.
  // In the later case, join the thread and
  int get_idle_tid()
  {
    int tid = (last_check_tid_ == thread_num_ - 1) ? 0 : last_check_tid_ + 1;
    std::chrono::microseconds span(100);

    // Loop until an idle thread is found
    while (true)
    {
      // Return immediately if a thread that has not been given a job is found
      if (!thread_futs_[tid].valid())
      {
        last_check_tid_ = tid;
        return tid;
      }

      // If no such thread is found, wait for the current thread to finish its job
      auto stat = thread_futs_[tid].wait_for(span);

      if (stat == std::future_status::ready)
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

  bool applyFilterThread(int tid, const std::string &grid_id, LeafDict &leaves) 
  {
    // For debug, measure the exe time
    struct timeval start, end;

    gettimeofday(&start, NULL);
    applyFilter(processing_inputs_[tid], grid_id, leaves);

    // For debug
    int size = processing_inputs_[tid]->size();
    // End

    // Clean the processed input cloud
    processing_inputs_[tid].reset();
    gettimeofday(&end, NULL);

    // Save the processing time in milliseconds
    std::ostringstream val;

    val << "Thread " << tid << " exe time = " << timeDiff(start, end) << " input size = " << size << std::endl;

    test_mtx_.lock();
    test_file_ << val.str();
    test_mtx_.unlock();    

    return true;
  }

  /** \brief Filter cloud and initializes voxel structure.
   * \param[out] output cloud containing centroids of voxels containing a sufficient number of points
   */
  void applyFilter(const PointCloudConstPtr &input, const std::string &grid_id, LeafDict &leaves);

  void updateVoxelCentroids(const Leaf &leaf, PointCloud &voxel_centroids) const;

  void updateLeaf(const PointT &point, const int &centroid_size, Leaf &leaf) const;

  void computeLeafParams(const Eigen::Vector3d &pt_sum, Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> &eigensolver, Leaf &leaf) const;

  LeafID getLeafID(const std::string &grid_id, const PointT &point, const BoundingBox &bbox) const;

  // Debug functions
  // Cut the full path to only get the file name 
  std::string cutFname(const std::string& input)
  {
    auto last_slash = input.rfind("/");

    return input.substr(last_slash + 1);
  }

  void leafToPCD(const std::string& leaf_name, const LeafDict& grid_leaf)
  {
    auto fname = cutFname(leaf_name);

    test_file_ << __FILE__ << "::" << __LINE__ << "::" << __func__ << "::save to = " << fname << " size = " << grid_leaf.size() << std::endl;

    if (grid_leaf.size() == 0)
    {
      test_file_ << __FILE__ << "::" << __LINE__ << "::" << __func__ << "::empty leaf = " << leaf_name << std::endl;
      return;
    }

    pcl::PointCloud<PointT> out_cloud;

    out_cloud.reserve(grid_leaf.size());

    for (auto& p : grid_leaf)
    {
      PointT op;

      op.x = p.second.centroid[0];
      op.y = p.second.centroid[1];
      op.z = p.second.centroid[2];

      out_cloud.push_back(op);
    }

    pcl::io::savePCDFileBinary("/home/anh/Work/autoware/" + fname, out_cloud);
  }

  /** \brief Minimum points contained with in a voxel to allow it to be usable. */
  int min_points_per_voxel_;

  /** \brief Minimum allowable ratio between eigenvalues to prevent singular covariance matrices. */
  double min_covar_eigvalue_mult_;

  /** \brief Voxel structure containing all leaf nodes (includes voxels with less than a sufficient number of points). */
  LeafDict leaves_;

  /** \brief Point cloud containing centroids of voxels containing at least minimum number of points. */
  std::map<std::string, LeafDict> grid_leaves_;

  /** \brief Indices of leaf structures associated with each point in \ref voxel_centroids_ (used for searching). */
  std::vector<LeafID> leaf_indices_;

  /** \brief KdTree generated using \ref voxel_centroids_ (used for searching). */
  pcl::KdTreeFLANN<PointT> kdtree_;

  PointCloudPtr voxel_centroids_ptr_;

  // Thread pooling, for parallel processing
  int thread_num_;
  std::vector<std::future<bool>> thread_futs_;
  std::vector<PointCloudConstPtr> processing_inputs_;
  int last_check_tid_;
  // For debug
  struct timeval all_start_, all_end_;
  std::fstream test_file_;
  std::mutex test_mtx_;
  // End
};
}  // namespace pclomp

#endif  // #ifndef PCL_MULTI_VOXEL_GRID_COVARIANCE_H_
