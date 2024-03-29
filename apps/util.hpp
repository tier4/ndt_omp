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

#ifndef NDT_OMP__APPS__UTIL_HPP_
#define NDT_OMP__APPS__UTIL_HPP_

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

pcl::PointCloud<pcl::PointXYZ>::Ptr load_pcd(const std::string& path) {
  // check if dir
  if(!std::filesystem::exists(path)) {
    std::cerr << "failed to find " << path << std::endl;
    std::exit(1);
  }
  if(std::filesystem::is_directory(path)) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcd(new pcl::PointCloud<pcl::PointXYZ>());
    for(const auto& file : glob(path)) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>());
      if(pcl::io::loadPCDFile(file, *tmp)) {
        std::cerr << "failed to load " << file << std::endl;
        std::exit(1);
      }
      *pcd += *tmp;
    }
    return pcd;
  } else {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcd(new pcl::PointCloud<pcl::PointXYZ>());
    if(pcl::io::loadPCDFile(path, *pcd)) {
      std::cerr << "failed to load " << path << std::endl;
      std::exit(1);
    }
    return pcd;
  }
}

std::vector<Eigen::Matrix4f> load_pose_list(const std::string& path) {
  /*
  timestamp,pose_x,pose_y,pose_z,quat_w,quat_x,quat_y,quat_z,twist_linear_x,twist_linear_y,twist_linear_z,twist_angular_x,twist_angular_y,twist_angular_z
  63.100010,81377.359702,49916.899866,41.232589,0.953768,0.000494,-0.007336,0.300453,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
  63.133344,81377.359780,49916.899912,41.232735,0.953769,0.000491,-0.007332,0.300452,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
  ...
  */
  std::ifstream ifs(path);
  std::string line;
  std::getline(ifs, line);  // skip header
  std::vector<Eigen::Matrix4f> pose_list;
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
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    pose.block<3, 3>(0, 0) = Eigen::Quaternionf(quat_w, quat_x, quat_y, quat_z).toRotationMatrix();
    pose.block<3, 1>(0, 3) = Eigen::Vector3f(pose_x, pose_y, pose_z);
    pose_list.push_back(pose);
  }
  return pose_list;
}

#endif  // NDT_OMP__APPS__UTIL_HPP_
