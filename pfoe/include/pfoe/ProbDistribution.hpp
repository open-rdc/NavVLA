#pragma once
#include <random>

namespace pfoe {

class ProbDistributions {
public:
  ProbDistributions() : gen_(rd_()) {}

  double uniformRand(double min, double max) {
    std::uniform_real_distribution<> ud(min, max);
    return ud(gen_);
  }

  int uniformRandInt(int min, int max) {
    std::uniform_int_distribution<> ud(min, max);
    return ud(gen_);
  }

private:
  std::random_device rd_;
  std::mt19937       gen_;
};

}  // namespace pfoe
