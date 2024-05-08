// Copyright 2024 Autoware Foundation
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

#ifndef PARTICLE_FILTER__PARTICLE_FILTER_HPP_
#define PARTICLE_FILTER__PARTICLE_FILTER_HPP_

#include <Eigen/Core>

namespace particle_filter {

struct Particle {
  Eigen::Matrix4f pose;
  float score;
  float weight;
};

struct ParticleFilterParams {
  int64_t num_particles;
  std::function<float(const Eigen::Matrix4f &)> score_function;
  Eigen::Matrix4f initial_pose;
  std::array<float, 6> covariance_diagonal;  // trans_x, trans_y, trans_z, angle_x, angle_y, angle_z (angle_z is ignored)
};

class ParticleFilter {
public:
  ParticleFilter(const ParticleFilterParams &params);

  void predict(const Eigen::Matrix4f &delta_pose);

  void update();

  void resample();

  bool is_converged() const;

  float effective_sample_size() const;

  const std::vector<Particle> &get_particles() const {
    return particles_;
  }

private:
  void add_randomness(Eigen::Matrix4f &pose, const std::array<float, 6> &covariance_diagonal, const float stddev_scale);

  ParticleFilterParams params_;
  std::vector<Particle> particles_;
};

}  // namespace particle_filter

#endif  // PARTICLE_FILTER__PARTICLE_FILTER_HPP_
