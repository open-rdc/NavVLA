#pragma once
#include <vector>
#include <cmath>

namespace pfoe {

struct Event {
  std::vector<float> observation;  // CLIP feature (dim=512)

  double likelihood(const Event& ref) const;
};

}  // namespace pfoe
