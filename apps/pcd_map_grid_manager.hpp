// Copyright 2024 TIER IV, Inc.
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

#ifndef NDT_OMP__APPS__PCD_MAP_GRID_MANAGER_HPP_
#define NDT_OMP__APPS__PCD_MAP_GRID_MANAGER_HPP_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

using AddPair = std::pair<std::string, pcl::PointCloud<pcl::PointXYZ>::Ptr>;
using AddResult = std::vector<AddPair>;
using RemoveResult = std::vector<std::string>;

class MapGridManager
{
public:
  explicit MapGridManager(const pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud)
  : target_cloud_(target_cloud)
  {
    for (const auto & point : target_cloud_->points) {
      const int x = static_cast<int>(std::ceil(point.x / resolution_));
      const int y = static_cast<int>(std::ceil(point.y / resolution_));
      const std::pair<int, int> key = std::make_pair(x, y);
      if (map_grid_.count(key) == 0) {
        map_grid_[key].reset(new pcl::PointCloud<pcl::PointXYZ>());
      }
      map_grid_[key]->points.push_back(point);
    }
  }

  std::pair<AddResult, RemoveResult> query(const Eigen::Matrix4f & pose)
  {
    // get around
    const float x = pose(0, 3);
    const float y = pose(1, 3);
    const int x_min = static_cast<int>(std::ceil((x - get_around_) / resolution_));
    const int x_max = static_cast<int>(std::ceil((x + get_around_) / resolution_));
    const int y_min = static_cast<int>(std::ceil((y - get_around_) / resolution_));
    const int y_max = static_cast<int>(std::ceil((y + get_around_) / resolution_));
    std::vector<std::pair<int, int>> curr_keys;
    for (int x = x_min; x <= x_max; x++) {
      for (int y = y_min; y <= y_max; y++) {
        if (map_grid_.count(std::make_pair(x, y)) == 0) {
          continue;
        }
        curr_keys.emplace_back(std::make_pair(x, y));
      }
    }

    // get remove keys
    RemoveResult remove_result;
    for (const auto & key : held_keys_) {
      if (std::find(curr_keys.begin(), curr_keys.end(), key) == curr_keys.end()) {
        remove_result.push_back(to_string_key(key));
      }
    }

    // get add keys
    AddResult add_result;
    for (const auto & key : curr_keys) {
      if (std::find(held_keys_.begin(), held_keys_.end(), key) == held_keys_.end()) {
        add_result.push_back(std::make_pair(to_string_key(key), map_grid_[key]));
      }
    }

    held_keys_ = curr_keys;
    return std::make_pair(add_result, remove_result);
  }

private:
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_;
  const float resolution_ = 20.0;   // [m]
  const float get_around_ = 100.0;  // [m]
  std::map<std::pair<int, int>, pcl::PointCloud<pcl::PointXYZ>::Ptr> map_grid_;
  std::vector<std::pair<int, int>> held_keys_;

  static std::string to_string_key(const std::pair<int, int> & key)
  {
    return std::to_string(key.first) + "_" + std::to_string(key.second);
  }
};

#endif  // NDT_OMP__APPS__PCD_MAP_GRID_MANAGER_HPP_
