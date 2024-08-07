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

#ifndef NDT_OMP__APPS__TIMER_HPP_
#define NDT_OMP__APPS__TIMER_HPP_

#include <chrono>
#include <iomanip>
#include <string>

class Timer
{
public:
  void start() { start_time_ = std::chrono::steady_clock::now(); }

  [[nodiscard]] double elapsed_milliseconds() const
  {
    const auto end_time = std::chrono::steady_clock::now();
    const auto elapsed_time = end_time - start_time_;
    const double microseconds = static_cast<double>(
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count());
    return microseconds / 1000.0;
  }

private:
  std::chrono::steady_clock::time_point start_time_;
};

#endif  //  NDT_OMP__APPS__TIMER_HPP_
