#include "pfoe/ParticleFilter.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>

namespace pfoe {

ParticleFilter::ParticleFilter(int num)
{
  double w = 1.0 / num;
  Particle p(w);
  for (int i = 0; i < num; ++i) {
    particles.push_back(p);
    retro_particles.push_back(p);
  }
}

void ParticleFilter::reset(Episode* ep)
{
  double w = 1.0 / static_cast<double>(particles.size());
  for (auto& p : particles) {
    p.time   = prob_.uniformRandInt(1, ep->size() - 1);
    p.weight = w;
  }
}

void ParticleFilter::scatter(Episode* ep, int center, int spread)
{
  const int max_time = ep->size() - 1;
  int c = center;
  if (c < 1)        c = 1;
  if (c > max_time) c = max_time;
  int lo = c - spread;
  int hi = c + spread;
  if (lo < 1)        lo = 1;
  if (hi > max_time) hi = max_time;

  const double w = 1.0 / static_cast<double>(particles.size());
  for (auto& p : particles) {
    p.time   = prob_.uniformRandInt(lo, hi);
    p.weight = w;
  }
}

void ParticleFilter::set_observation_temperature(double beta)
{
  beta_ = beta;
}

void ParticleFilter::set_motion_model(double p_back, double p_stay,
                                      double p_forward, double p_skip)
{
  // renormalize to a proper probability distribution
  double s = p_back + p_stay + p_forward + p_skip;
  if (s <= 0.0) return;  // keep defaults on invalid input
  p_back_    = p_back    / s;
  p_stay_    = p_stay    / s;
  p_forward_ = p_forward / s;
  p_skip_    = p_skip    / s;
}

void ParticleFilter::update(Episode* ep, double weight_sum_thresh, int retro_steps)
{
  Event* cur = ep->current();
  moveAndBayes(ep, &particles, cur);
  double sum = sumWeight(&particles);
  if (sum > weight_sum_thresh) {
    resampling(&particles);
  } else {
    retrospectiveFilter(ep, &particles, retro_steps);
  }
}

int ParticleFilter::best_time_idx() const
{
  // MAP estimate = mode of the particle distribution (weight-weighted
  // histogram over time). After resampling all weights are equal, so this
  // reduces to the most populated time bin; it is well-defined regardless
  // of whether resampling has just run (unlike argmax over equal weights).
  std::unordered_map<int, double> hist;
  for (const auto& p : particles) hist[p.time] += p.weight;

  int    best_t = 1;
  double best_w = -1.0;
  for (const auto& kv : hist) {
    if (kv.second > best_w) {
      best_w = kv.second;
      best_t = kv.first;
    }
  }
  return best_t;
}

void ParticleFilter::moveAndBayes(Episode* ep,
                                   std::vector<Particle>* ps,
                                   Event* cur)
{
  const int max_time = ep->size() - 1;  // particles live in [1, N-1]

  // --- Method C: stochastic motion model over the trajectory index ---
  // The prediction step no longer advances unconditionally; a particle may
  // step -1 (recover from overshoot), stay (handles a stationary robot),
  // +1, or +2 (fast motion). The image stream decides which hypothesis
  // survives via the Bayes step. The backward step prevents the forward
  // ratchet that otherwise pins a stationary estimate at the end.
  for (auto& p : *ps) {
    const double r = prob_.uniformRand(0.0, 1.0);
    int step;
    if (r < p_back_)                               step = -1;
    else if (r < p_back_ + p_stay_)                step = 0;
    else if (r < p_back_ + p_stay_ + p_forward_)   step = 1;
    else                                           step = 2;
    p.time += step;
    if (p.time > max_time) p.time = max_time;
    if (p.time < 1)        p.time = 1;
  }

  // --- Method B: temperature-scaled observation model exp(beta * cos) ---
  for (auto& p : *ps)
    p.weight *= std::exp(beta_ * ep->at(p.time)->similarity(*cur));
}

void ParticleFilter::retrospectiveFilter(Episode* ep,
                                          std::vector<Particle>* ps,
                                          int step)
{
  std::cerr << "             !!!!!!!!!!!!!!!!" << std::endl;
  for (auto& p : *ps) {
    p.time   = prob_.uniformRandInt(1, ep->size() - 1);
    p.weight = 1.0 / static_cast<double>(ps->size());
  }

  int count = 0;
  const int avail = ep->recent_size();
  for (int i = 0; i < step; ++i) {
    if (i + 1 > avail) break;

    Event* cur = ep->recent(i + 1);
    for (auto& p : *ps) {
      p.weight *= std::exp(beta_ * ep->at(p.time)->similarity(*cur));
      p.weight += 0.000001;
    }
    resampling(ps);
    // time shift
    for (auto& p : *ps) {
      p.time--;
      if (p.time == 0) p.time = 1;
    }
    count++;
  }

  for (auto& p : *ps) {
    p.time += count;
    if (p.time >= ep->size()) p.time = ep->size() - 1;
  }
}

void ParticleFilter::resampling(std::vector<Particle>* ps)
{
  std::vector<Particle> prev;
  std::shuffle(ps->begin(), ps->end(), std::mt19937(std::random_device{}()));

  double sum_weight = 0.0;
  int    num        = static_cast<int>(ps->size());
  for (int i = 0; i < num; ++i) {
    if (ps->at(i).weight < 0.0000001) continue;

    // weight is changed to the accumulated value
    ps->at(i).weight += sum_weight;
    sum_weight = ps->at(i).weight;
    prev.push_back(ps->at(i));
  }
  if (prev.empty()) return;

  double           step  = sum_weight / num;
  std::vector<int> choice(num);
  double           accum = sum_weight * prob_.uniformRand(0.0, 1.0) / num;
  int              j     = 0;
  int              pn    = static_cast<int>(prev.size());
  for (int i = 0; i < num; ++i) {
    if (prev[j].weight <= accum) j++;
    if (j == pn) j--;

    accum    += step;
    choice[i] = j;
  }

  for (int i = 0; i < num; ++i) {
    int k            = choice[i];
    ps->at(i)        = prev[k];
    ps->at(i).weight = 1.0 / num;
  }
}

double ParticleFilter::sumWeight(const std::vector<Particle>* ps) const
{
  double sum = 0.0;
  for (const auto& p : *ps) sum += p.weight;
  return sum;
}

}  // namespace pfoe
