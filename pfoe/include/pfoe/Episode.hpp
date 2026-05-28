#pragma once
#include "Event.hpp"
#include <vector>
#include <string>

namespace pfoe {

class Episode {
public:
  Episode(int backtrack, double discount);

  int    size() const;
  Event* at(int i);       // 1-indexed
  Event* current();

  void push_back(const Event& e);

  static Episode load(const std::string& emb_path,
                      const std::string& prompt_path,
                      int backtrack, double discount);

  std::vector<std::string> instructions;  // [N] from traj_prompt.txt

private:
  std::vector<Event> events_;
  int                backtrack_threshold_;
  double             discount_rate_;
};

}  // namespace pfoe
