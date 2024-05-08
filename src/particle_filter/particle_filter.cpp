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

#include "particle_filter/particle_filter.hpp"
#include <random>
#include <Eigen/Geometry>

namespace particle_filter {

ParticleFilter::ParticleFilter(const ParticleFilterParams &params) {
  params_ = params;
  particles_.resize(params.num_particles);
  for(Particle &particle : particles_) {
    particle.weight = 1.0f / params.num_particles;
    particle.pose = params.initial_pose;
    add_randomness(particle.pose, params.covariance_diagonal, 1.0f);
  }
}

void ParticleFilter::predict(const Eigen::Matrix4f &delta_pose) {
  for(Particle &particle : particles_) {
    particle.pose = particle.pose * delta_pose;
    add_randomness(particle.pose, params_.covariance_diagonal, 0.1f);
  }
}

void ParticleFilter::update() {
  float sum_weight = 0.0f;
  for(Particle &particle : particles_) {
    params_.score_function(particle);
    particle.weight *= std::exp(particle.score);
    sum_weight += particle.weight;
  }

  for(Particle &particle : particles_) {
    particle.weight /= sum_weight;
  }
}

void ParticleFilter::resample() {
  const float ess = effective_sample_size();
  if(ess > 0.1f * params_.num_particles) {
    return;
  }
  const float step = 1.0f / params_.num_particles;
  std::mt19937_64 engine(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0f, step);
  const float start = dist(engine);
  int64_t index = 1;
  float sum = particles_[0].weight;
  float new_sum = 0.0f;
  std::vector<Particle> new_particles(params_.num_particles);
  for(int64_t i = 0; i < params_.num_particles; i++) {
    const float target = start + i * step;
    while(sum < target && index < params_.num_particles) {
      sum += particles_[index].weight;
      index++;
    }
    new_particles[i] = particles_[index - 1];
    new_sum += new_particles[i].weight;
  }
  particles_ = new_particles;
  for(Particle &particle : particles_) {
    particle.weight /= new_sum;
  }
}

bool ParticleFilter::is_converged() const {
  return effective_sample_size() > 0.9f * params_.num_particles;
}

float ParticleFilter::effective_sample_size() const {
  float sum = 0.0f;
  for(const Particle &particle : particles_) {
    sum += particle.weight * particle.weight;
  }
  return 1.0f / sum;
}

void ParticleFilter::add_randomness(Eigen::Matrix4f &pose, const std::array<float, 6> &covariance_diagonal, const float stddev_scale) {
  std::mt19937_64 engine(std::random_device{}());
  std::vector<std::normal_distribution<float>> dist_vec;
  for(float diag : covariance_diagonal) {
    dist_vec.emplace_back(0.0f, stddev_scale * std::sqrt(diag));
  }
  // trans
  pose(0, 3) += dist_vec[0](engine);
  pose(1, 3) += dist_vec[1](engine);
  pose(2, 3) += dist_vec[2](engine);

  // angle
  const float angle_x = dist_vec[3](engine);
  const float angle_y = dist_vec[4](engine);
  const float angle_z = dist_vec[5](engine);
  const Eigen::Matrix3f rot_x = Eigen::AngleAxisf(angle_x, Eigen::Vector3f::UnitX()).toRotationMatrix();
  const Eigen::Matrix3f rot_y = Eigen::AngleAxisf(angle_y, Eigen::Vector3f::UnitY()).toRotationMatrix();
  const Eigen::Matrix3f rot_z = Eigen::AngleAxisf(angle_z, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  pose.block<3, 3>(0, 0) = rot_z * rot_y * rot_x * pose.block<3, 3>(0, 0);
}

};  // namespace particle_filter
