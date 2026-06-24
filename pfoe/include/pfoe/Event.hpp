#pragma once
#include <vector>
#include <cmath>

namespace pfoe {

struct Event {
  std::vector<float> observation;  // CLIP feature (dim=512)

  // Raw cosine similarity in [-1, 1] between this observation and ref.
  // The temperature-scaled observation likelihood exp(beta * similarity)
  // is applied by the particle filter (Method B).
  double similarity(const Event& ref) const;
};

}  // namespace pfoe
