#pragma once

#include "pfoe/Episode.hpp"
#include "pfoe/ParticleFilter.hpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/int32.hpp>

#include <memory>

namespace pfoe {

class PFoENode : public rclcpp::Node {
public:
  PFoENode();

private:
  void feature_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);

  std::unique_ptr<Episode>                                          episode_;
  std::unique_ptr<ParticleFilter>                                   pf_;
  double                                                            weight_sum_thresh_;
  int                                                               retro_steps_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr feature_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr               prompt_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr                time_idx_pub_;
};

}  // namespace pfoe
