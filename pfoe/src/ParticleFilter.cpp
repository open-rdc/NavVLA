#include "pfoe/ParticleFilter.hpp"
#include <iostream>
#include <algorithm>
#include <random>

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
  int sz = ep->size();
  int n  = static_cast<int>(particles.size());
  for (int i = 0; i < n; ++i) {
    particles[i].time   = prob_.uniformRandInt(1, sz > 1 ? sz - 1 : 1);
    particles[i].weight = w;
  }
}

void ParticleFilter::update(Episode* ep)
{
  Event* cur = ep->current();
  moveAndBayes(ep, &particles, cur);
  double sum = sumWeight(&particles);
  std::cerr << "sum of weights: " << sum << std::endl;
  if (sum > 0.001) {
    resampling(&particles);
  } else {
    retrospectiveFilter(ep, &particles, 20);
  }
}

int ParticleFilter::best_time_idx() const
{
  int    best_t = 1;
  double best_w = -1.0;
  for (const auto& p : particles) {
    if (p.weight > best_w) {
      best_w = p.weight;
      best_t = p.time;
    }
  }
  return best_t;
}

void ParticleFilter::moveAndBayes(Episode* ep,
                                   std::vector<Particle>* ps,
                                   Event* cur)
{
  int ep_sz = ep->size();
  for (auto& p : *ps) {
    p.time++;
    if (p.time > ep_sz) p.time = ep_sz;
  }
  for (auto& p : *ps) {
    p.weight *= ep->at(p.time)->likelihood(*cur);
  }
}

void ParticleFilter::retrospectiveFilter(Episode* ep,
                                          std::vector<Particle>* ps,
                                          int step)
{
  std::cerr << "             !!!!!!!!!!!!!!!!" << std::endl;
  int ep_sz = ep->size();
  for (auto& p : *ps) {
    p.time   = prob_.uniformRandInt(1, ep_sz > 1 ? ep_sz - 1 : 1);
    p.weight = 1.0 / static_cast<double>(ps->size());
  }

  int count = 0;
  for (int i = 0; i < step; ++i) {
    if (ep_sz - i <= 1) break;
    Event* cur = ep->at(ep_sz - i);
    for (auto& p : *ps) {
      p.weight *= ep->at(p.time)->likelihood(*cur);
      p.weight += 0.000001;
    }
    resampling(ps);
    for (auto& p : *ps) {
      p.time--;
      if (p.time == 0) p.time = 1;
    }
    count++;
  }

  for (auto& p : *ps) {
    p.time += count;
    if (p.time >= ep_sz) p.time = ep_sz - 1;
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
    ps->at(i).weight += sum_weight;
    sum_weight = ps->at(i).weight;
    prev.push_back(ps->at(i));
  }
  if (prev.empty()) return;

  double step  = sum_weight / num;
  double accum = sum_weight * prob_.uniformRand(0.0, 1.0) / num;
  int    j     = 0;

  std::vector<int> choice(num);
  for (int i = 0; i < num; ++i) {
    while (j < static_cast<int>(prev.size()) - 1 && prev[j].weight <= accum)
      j++;
    choice[i] = j;
    accum += step;
  }

  for (int i = 0; i < num; ++i) {
    ps->at(i)        = prev[choice[i]];
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
