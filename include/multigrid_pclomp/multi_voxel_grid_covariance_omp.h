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
namespace pclomp
{
  /** \brief A searchable voxel structure containing the mean and covariance of the data.
    * \note For more information please see
    * <b>Magnusson, M. (2009). The Three-Dimensional Normal-Distributions Transform —
    * an Efficient Representation for Registration, Surface Analysis, and Loop Detection.
    * PhD thesis, Orebro University. Orebro Studies in Technology 36</b>
    * \author Brian Okorn (Space and Naval Warfare Systems Center Pacific)
    */
  template<typename PointT>
  class MultiVoxelGridCovariance : public pcl::VoxelGrid<PointT>
  {
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
      typedef pcl::shared_ptr< pcl::VoxelGrid<PointT> > Ptr;
      typedef pcl::shared_ptr< const pcl::VoxelGrid<PointT> > ConstPtr;
#else
      typedef boost::shared_ptr< pcl::VoxelGrid<PointT> > Ptr;
      typedef boost::shared_ptr< const pcl::VoxelGrid<PointT> > ConstPtr;
#endif

      /** \brief Simple structure to hold a centroid, covariance and the number of points in a leaf.
        * Inverse covariance, eigen vectors and eigen values are precomputed. */
      struct Leaf
      {
        /** \brief Constructor.
         * Sets \ref nr_points, \ref icov_, \ref mean_ and \ref evals_ to 0 and \ref cov_ and \ref evecs_ to the identity matrix
         */
        Leaf () :
          nr_points (0),
          mean_ (Eigen::Vector3d::Zero ()),
          centroid (),
          cov_ (Eigen::Matrix3d::Identity ()),
          icov_ (Eigen::Matrix3d::Zero ()),
          evecs_ (Eigen::Matrix3d::Identity ()),
          evals_ (Eigen::Vector3d::Zero ())
        {
        }

        /** \brief Get the voxel covariance.
          * \return covariance matrix
          */
        Eigen::Matrix3d
        getCov () const
        {
          return (cov_);
        }

        /** \brief Get the inverse of the voxel covariance.
          * \return inverse covariance matrix
          */
        Eigen::Matrix3d
        getInverseCov () const
        {
          return (icov_);
        }

        /** \brief Get the voxel centroid.
          * \return centroid
          */
        Eigen::Vector3d
        getMean () const
        {
          return (mean_);
        }

        /** \brief Get the eigen vectors of the voxel covariance.
          * \note Order corresponds with \ref getEvals
          * \return matrix whose columns contain eigen vectors
          */
        Eigen::Matrix3d
        getEvecs () const
        {
          return (evecs_);
        }

        /** \brief Get the eigen values of the voxel covariance.
          * \note Order corresponds with \ref getEvecs
          * \return vector of eigen values
          */
        Eigen::Vector3d
        getEvals () const
        {
          return (evals_);
        }

        /** \brief Get the number of points contained by this voxel.
          * \return number of points
          */
        int
        getPointCount () const
        {
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
        bool operator < (const LeafID& rhs) const {
          if (parent_grid_id < rhs.parent_grid_id) {
            return true;
          }
          if (parent_grid_id > rhs.parent_grid_id) {
            return false;
          }
          if (leaf_index < rhs.leaf_index) {
            return true;
          }
          if (leaf_index > rhs.leaf_index) {
            return false;
          }
          return false;
        }
      };

      /** \brief Pointer to MultiVoxelGridCovariance leaf structure */
      typedef Leaf* LeafPtr;

      /** \brief Const pointer to MultiVoxelGridCovariance leaf structure */
      typedef const Leaf* LeafConstPtr;

      typedef std::map<LeafID, Leaf> Map;

      struct VoxelGridInfo {
        PointCloud voxel_centroids;
        Map leaves;
        std::vector<LeafID> leaf_indices;
        void clear(){
          voxel_centroids.clear();
          leaves.clear();
          leaf_indices.clear();
        }
      };

      struct BoundingBox
      {
        Eigen::Vector4i max;
        Eigen::Vector4i min;
        Eigen::Vector4i div_mul;
      };

    public:

      /** \brief Constructor.
       * Sets \ref leaf_size_ to 0 and \ref searchable_ to false.
       */
      MultiVoxelGridCovariance () :
        searchable_ (true),
        min_points_per_voxel_ (6),
        min_covar_eigvalue_mult_ (0.01),
        leaves_ (),
        voxel_grid_info_dict_ (),
        voxel_centroids_leaf_indices_ (),
        kdtree_ (),
        voxel_grid_info_all_ ()
      {
        leaf_size_.setZero ();
        min_b_.setZero ();
        max_b_.setZero ();
        filter_name_ = "MultiVoxelGridCovariance";
      }

      /** \brief Initializes voxel structure.
       * \param[in] searchable flag if voxel structure is searchable, if true then kdtree is built
       */
      inline void
      setInputCloudAndFilter (const PointCloudConstPtr &cloud, const std::string &grid_id, bool searchable = false)
      {
        searchable_ = searchable;
        VoxelGridInfo voxel_grid_info;
        applyFilter (cloud, grid_id, voxel_grid_info);

        voxel_grid_info_dict_[grid_id] = voxel_grid_info;
      }

      inline void 
      removeCloud (const std::string &grid_id)
      {
        voxel_grid_info_dict_.erase(grid_id);
      }

      inline VoxelGridInfo
      concatVoxelGridInfoDict(std::map<std::string, VoxelGridInfo> &input) const
      {
        VoxelGridInfo output;
        for (const auto &kv: input)
        {
          output.voxel_centroids += kv.second.voxel_centroids;
          output.leaves.insert(kv.second.leaves.begin(), kv.second.leaves.end());
          output.leaf_indices.insert(output.leaf_indices.end(), kv.second.leaf_indices.begin(), kv.second.leaf_indices.end());
        }
        return output;
      }

      inline void 
      createKdtree ()
      {
        voxel_grid_info_all_.clear();
        voxel_grid_info_all_ = concatVoxelGridInfoDict(voxel_grid_info_dict_);

        leaves_ = voxel_grid_info_all_.leaves;
        voxel_centroids_leaf_indices_ = voxel_grid_info_all_.leaf_indices;

        if (voxel_grid_info_all_.voxel_centroids.size() > 0)
        {
          kdtree_.setInputCloud (voxel_grid_info_all_.voxel_centroids.makeShared());
        }
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
      int
      radiusSearch (const PointT &point, double radius, std::vector<LeafConstPtr> &k_leaves,
                    std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const
      {
        k_leaves.clear ();

        // Check if kdtree has been built
        if (!searchable_)
        {
          PCL_WARN ("%s: Not Searchable", this->getClassName ().c_str ());
          return 0;
        }

        // Find neighbors within radius in the occupied voxel centroid cloud
        std::vector<int> k_indices;
        int k = kdtree_.radiusSearch (point, radius, k_indices, k_sqr_distances, max_nn);

        // Find leaves corresponding to neighbors
        k_leaves.reserve (k);
        for (std::vector<int>::iterator iter = k_indices.begin (); iter != k_indices.end (); iter++)
        {
          auto leaf = leaves_.find(voxel_centroids_leaf_indices_[*iter]);
          if (leaf == leaves_.end()) {
            std::cerr << "error : could not find the leaf corresponding to the voxel" << std::endl;
            std::cin.ignore(1);
          }
          k_leaves.push_back (&(leaf->second));
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
      inline int
      radiusSearch (const PointCloud &cloud, int index, double radius,
                    std::vector<LeafConstPtr> &k_leaves, std::vector<float> &k_sqr_distances,
                    unsigned int max_nn = 0) const
      {
        if (index >= static_cast<int> (cloud.points.size ()) || index < 0)
          return (0);
        return (radiusSearch (cloud.points[index], radius, k_leaves, k_sqr_distances, max_nn));
      }

      PointCloud getVoxelPCD () const
      {
        return voxel_grid_info_all_.voxel_centroids;
      }

  		std::vector<std::string> getCurrentMapIDs() const
      {
        std::vector<std::string> output{};
        for (const auto &element: voxel_grid_info_dict_) {
          output.push_back(element.first);
        }
        return output;
      }

    protected:

      /** \brief Filter cloud and initializes voxel structure.
       * \param[out] output cloud containing centroids of voxels containing a sufficient number of points
       */
      void applyFilter (const PointCloudConstPtr &input, const std::string &grid_id, VoxelGridInfo &voxel_grid_info) const;

      void updateVoxelCentroids (const Leaf &leaf, PointCloud &voxel_centroids) const;

      void updateLeaf (const PointT &point, const int &centroid_size, Leaf &leaf) const;

      void computeLeafParams (const Eigen::Vector3d &pt_sum,
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> &eigensolver,
        Leaf &leaf) const;

      LeafID getLeafID (const std::string &grid_id, const PointT &point, const BoundingBox &bbox) const;

      /** \brief Flag to determine if voxel structure is searchable. */
      bool searchable_;

      /** \brief Minimum points contained with in a voxel to allow it to be usable. */
      int min_points_per_voxel_;

      /** \brief Minimum allowable ratio between eigenvalues to prevent singular covariance matrices. */
      double min_covar_eigvalue_mult_;

      /** \brief Voxel structure containing all leaf nodes (includes voxels with less than a sufficient number of points). */
	    Map leaves_;

      /** \brief Point cloud containing centroids of voxels containing at least minimum number of points. */
      std::map<std::string, VoxelGridInfo> voxel_grid_info_dict_;

      /** \brief Indices of leaf structures associated with each point in \ref voxel_centroids_ (used for searching). */
      std::vector<LeafID> voxel_centroids_leaf_indices_;

      /** \brief KdTree generated using \ref voxel_centroids_ (used for searching). */
      pcl::KdTreeFLANN<PointT> kdtree_;

      VoxelGridInfo voxel_grid_info_all_;
  };
}

#endif  //#ifndef PCL_MULTI_VOXEL_GRID_COVARIANCE_H_
