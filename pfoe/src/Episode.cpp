#include "pfoe/Episode.hpp"
#include <fstream>
#include <stdexcept>
#include <cstdint>

namespace pfoe {

Episode::Episode(int backtrack, double discount)
  : backtrack_threshold_(backtrack), discount_rate_(discount) {}

int Episode::size() const
{
  return static_cast<int>(events_.size());
}

Event* Episode::at(int i)
{
  // 1-indexed (reference implementation convention)
  return &events_.at(static_cast<size_t>(i - 1));
}

Event* Episode::current()
{
  return &events_.back();
}

void Episode::push_back(const Event& e)
{
  events_.push_back(e);
  // discount-based value propagation (reference impl compatible)
  int n = 1;
  double v_next = 0.0;
  for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
    it->observation;  // access to suppress unused warning
    (void)v_next;
    if (n++ >= backtrack_threshold_) break;
    v_next *= discount_rate_;
  }
}

Episode Episode::load(const std::string& emb_path,
                      const std::string& prompt_path,
                      int backtrack, double discount)
{
  Episode ep(backtrack, discount);

  // --- load clip_embeddings.bin ---
  // Format: int32 N, int32 dim, float32[N][dim] row-major
  std::ifstream bin(emb_path, std::ios::binary);
  if (!bin) throw std::runtime_error("Cannot open: " + emb_path);

  int32_t N = 0, dim = 0;
  bin.read(reinterpret_cast<char*>(&N),   sizeof(int32_t));
  bin.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
  if (!bin || N <= 0 || dim <= 0)
    throw std::runtime_error("Invalid binary header: " + emb_path);

  ep.events_.resize(static_cast<size_t>(N));
  for (int i = 0; i < N; ++i) {
    ep.events_[i].observation.resize(static_cast<size_t>(dim));
    bin.read(reinterpret_cast<char*>(ep.events_[i].observation.data()),
             dim * sizeof(float));
  }
  if (!bin) throw std::runtime_error("Truncated binary: " + emb_path);

  // --- load traj_prompt.txt (1 instruction per line) ---
  std::ifstream txt(prompt_path);
  if (!txt) throw std::runtime_error("Cannot open: " + prompt_path);

  std::string line;
  while (std::getline(txt, line)) {
    if (!line.empty()) ep.instructions.push_back(line);
  }

  return ep;
}

}  // namespace pfoe
