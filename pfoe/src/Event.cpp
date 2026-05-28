#include "pfoe/Event.hpp"
#include <cmath>

namespace pfoe {

double Event::likelihood(const Event& ref) const
{
  const auto& a = observation;
  const auto& b = ref.observation;
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  double cos_sim = dot / (std::sqrt(na * nb) + 1e-8);
  // cos_sim ∈ [-1,1] → non-negative likelihood
  return std::exp(cos_sim + 1.0);
}

}  // namespace pfoe
