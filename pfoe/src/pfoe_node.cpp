#include "pfoe/PFoENode.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace pfoe {

PFoENode::PFoENode(): Node("pfoe_node",
    rclcpp::NodeOptions()
      .allow_undeclared_parameters(true)
      .automatically_declare_parameters_from_overrides(true))
{
  const std::vector<std::string> required = {
      "episode_data_dir",
      "traj_name",
      "num_particles",
      "weight_sum_thresh",
      "retro_steps",
      "backtrack",
      "discount",
  };
  for (const auto& name : required) {
    if (!has_parameter(name)) {
      RCLCPP_ERROR(get_logger(),
                   "Required parameter not provided via YAML: %s",
                   name.c_str());
      throw std::runtime_error("Missing required parameter: " + name);
    }
  }

  const std::string data_dir  = get_parameter("episode_data_dir").as_string();
  const std::string traj_name = get_parameter("traj_name").as_string();
  const int    num_particles  = get_parameter("num_particles").as_int();
  const int    backtrack      = get_parameter("backtrack").as_int();
  const double discount       = get_parameter("discount").as_double();
  weight_sum_thresh_          = get_parameter("weight_sum_thresh").as_double();
  retro_steps_                = get_parameter("retro_steps").as_int();
  scatter_spread_             = get_parameter("scatter_spread").as_int();

  pf_ = std::make_unique<ParticleFilter>(num_particles);

  const double obs_temperature = get_parameter("obs_temperature").as_double();
  const double p_back          = get_parameter("p_back").as_double();
  const double p_stay          = get_parameter("p_stay").as_double();
  const double p_forward       = get_parameter("p_forward").as_double();
  const double p_skip          = get_parameter("p_skip").as_double();
  pf_->set_observation_temperature(obs_temperature);
  pf_->set_motion_model(p_back, p_stay, p_forward, p_skip);

  std::string emb_path    = data_dir + "/" + traj_name + "/clip_embeddings.bin";
  std::string prompt_path = data_dir + "/" + traj_name + "/traj_prompt.txt";

  episode_ = std::make_unique<Episode>(Episode::load(emb_path, prompt_path, backtrack, discount));
  episode_->set_live_capacity(static_cast<size_t>(std::max(retro_steps_ + 1, 64)));
  pf_->reset(episode_.get());

  feature_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>("/image_feature", 10, [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) { feature_callback(msg); });
  set_time_sub_ = create_subscription<std_msgs::msg::Int32>("/pfoe/set_time_idx", 10, [this](const std_msgs::msg::Int32::SharedPtr msg) { set_time_callback(msg); });
  manual_sub_ = create_subscription<std_msgs::msg::Bool>("/pfoe/manual", 10, [this](const std_msgs::msg::Bool::SharedPtr msg) { manual_callback(msg); });

  prompt_pub_    = create_publisher<std_msgs::msg::String>("/prompt",        10);
  time_idx_pub_  = create_publisher<std_msgs::msg::Int32>("/pfoe/time_idx",  10);
  particles_pub_ = create_publisher<std_msgs::msg::Int32MultiArray>("/pfoe/particles", 10);

  RCLCPP_INFO(get_logger(), "pfoe_node ready");
}

void PFoENode::feature_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  Event ev;
  ev.observation.assign(msg->data.begin(), msg->data.end());

  episode_->push_back(ev);
  if (episode_->size() < 2) return;

  int t_star;
  if (manual_) {
    t_star = manual_time_;
  } else {
    pf_->update(episode_.get(), weight_sum_thresh_, retro_steps_);
    t_star = pf_->best_time_idx();
  }

  publish_prompt(t_star);

  auto idx_msg = std_msgs::msg::Int32();
  idx_msg.data = t_star;
  time_idx_pub_->publish(idx_msg);

  publish_particles();
}

void PFoENode::set_time_callback(const std_msgs::msg::Int32::SharedPtr msg)
{
  const int center = msg->data;
  pf_->scatter(episode_.get(), center, scatter_spread_);
  manual_time_ = center;

  publish_prompt(center);

  auto idx_msg = std_msgs::msg::Int32();
  idx_msg.data = center;
  time_idx_pub_->publish(idx_msg);

  publish_particles();

  RCLCPP_INFO(get_logger(), "scatter particles around node %d (spread %d)", center, scatter_spread_);
}

void PFoENode::manual_callback(const std_msgs::msg::Bool::SharedPtr msg)
{
  manual_ = msg->data;
  RCLCPP_INFO(get_logger(), "manual mode: %s", manual_ ? "ON" : "OFF");
}

void PFoENode::publish_prompt(int t_star)
{
  if (episode_->instructions.empty()) return;

  int inst_idx = t_star - 1;
  if (inst_idx < 0) inst_idx = 0;
  if (inst_idx >= static_cast<int>(episode_->instructions.size()))
    inst_idx = static_cast<int>(episode_->instructions.size()) - 1;

  auto prompt_msg = std_msgs::msg::String();
  prompt_msg.data = episode_->instructions[inst_idx];
  prompt_pub_->publish(prompt_msg);
}

void PFoENode::publish_particles()
{
  auto particles_msg = std_msgs::msg::Int32MultiArray();
  particles_msg.data.reserve(pf_->particles.size());
  for (const auto& p : pf_->particles) particles_msg.data.push_back(p.time);
  particles_pub_->publish(particles_msg);
}

}  // namespace pfoe

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pfoe::PFoENode>());
  rclcpp::shutdown();
  return 0;
}
