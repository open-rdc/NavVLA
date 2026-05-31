#pragma once
#include "Event.hpp"
#include <deque>
#include <vector>
#include <string>

namespace pfoe {

class Episode {
public:
  Episode(int backtrack, double discount);

  int    size() const;            // reference trajectory length N
  Event* at(int i);                // 1-indexed reference event
  Event* current();                // latest live observation
  Event* recent(int i);            // 1-indexed: 1 = latest, 2 = previous, ...
  int    recent_size() const;
  bool   has_live() const;

  void   push_back(const Event& e);   // append live observation (sliding window)
  void   set_live_capacity(size_t k);

  static Episode load(const std::string& emb_path,
                      const std::string& prompt_path,
                      int backtrack, double discount);

  std::vector<std::string> instructions;  // [N] from traj_prompt.txt

private:
  std::vector<Event> events_;      // reference trajectory, fixed size N
  std::deque<Event>  live_;        // recent live observations
  size_t             live_capacity_ = 64;
  int                backtrack_threshold_;
  double             discount_rate_;
};

}  // namespace pfoe
