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
  void update(Episode* ep, double weight_sum_thresh, int retro_steps);
  int  best_time_idx() const;

  // Method B: sharpness (temperature) of the observation model exp(beta * cos).
  void set_observation_temperature(double beta);
  // Method C: stochastic motion model P(-1) / P(stay) / P(+1) / P(+2).
  // The backward step lets the estimate recover from overshoot so a
  // stationary robot does not ratchet forward. Renormalized internally.
  void set_motion_model(double p_back, double p_stay, double p_forward, double p_skip);

  std::vector<Particle> particles;
  std::vector<Particle> retro_particles;

private:
  ProbDistributions prob_;

  double beta_      = 50.0;  // observation sharpness               (Method B)
  double p_back_    = 0.25;  // P(time -= 1) — recover from overshoot (Method C)
  double p_stay_    = 0.50;  // P(time unchanged) — handles still
  double p_forward_ = 0.20;  // P(time += 1)
  double p_skip_    = 0.05;  // P(time += 2) — fast motion

  void   moveAndBayes(Episode* ep, std::vector<Particle>* ps, Event* cur);
  void   retrospectiveFilter(Episode* ep, std::vector<Particle>* ps, int step);
  void   resampling(std::vector<Particle>* ps);
  double sumWeight(const std::vector<Particle>* ps) const;
};

}  // namespace pfoe
