#pragma once

namespace pfoe {

struct Particle {
  explicit Particle(double w) : weight(w), time(0) {}
  double weight;
  int    time;  // 1-indexed
};

}  // namespace pfoe
