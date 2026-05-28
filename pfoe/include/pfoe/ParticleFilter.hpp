#pragma once
#include "Episode.hpp"
#include "Particle.hpp"
#include "ProbDistribution.hpp"
#include <vector>

namespace pfoe {

class ParticleFilter {
public:
  explicit ParticleFilter(int num);

  void reset(Episode* ep);
  void update(Episode* ep);
  int  best_time_idx() const;

  std::vector<Particle> particles;
  std::vector<Particle> retro_particles;

private:
  ProbDistributions prob_;

  void   moveAndBayes(Episode* ep, std::vector<Particle>* ps, Event* cur);
  void   retrospectiveFilter(Episode* ep, std::vector<Particle>* ps, int step);
  void   resampling(std::vector<Particle>* ps);
  double sumWeight(const std::vector<Particle>* ps) const;
};

}  // namespace pfoe
