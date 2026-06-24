#pragma once

#include "pfoe/Episode.hpp"
#include "pfoe/ParticleFilter.hpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include <std_msgs/msg/bool.hpp>

#include <memory>

namespace pfoe {

class PFoENode : public rclcpp::Node {
public:
  PFoENode();

private:
  void feature_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
  void set_time_callback(const std_msgs::msg::Int32::SharedPtr msg);
  void manual_callback(const std_msgs::msg::Bool::SharedPtr msg);
  void publish_prompt(int t_star);
  void publish_particles();

  std::unique_ptr<Episode>                                          episode_;
  std::unique_ptr<ParticleFilter>                                   pf_;
  double                                                            weight_sum_thresh_;
  int                                                               retro_steps_;
  int                                                               scatter_spread_;
  bool                                                              manual_      = false;
  int                                                               manual_time_ = 1;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr feature_sub_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr             set_time_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr              manual_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr               prompt_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr                time_idx_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr      particles_pub_;
};

}  // namespace pfoe
