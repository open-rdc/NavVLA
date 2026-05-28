#include "pfoe/Episode.hpp"
#include "pfoe/ParticleFilter.hpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/int32.hpp>

#include <memory>
#include <string>

namespace pfoe {

class PFoENode : public rclcpp::Node {
public:
  PFoENode() : Node("pfoe_node"), pf_(1000)
  {
    // --- parameters ---
    declare_parameter("episode_data_dir", std::string(""));
    declare_parameter("traj_name",        std::string("episode01"));
    declare_parameter("num_particles",    1000);
    declare_parameter("weight_sum_thresh",0.001);
    declare_parameter("retro_steps",      20);
    declare_parameter("backtrack",        1000);
    declare_parameter("discount",         0.99);

    const std::string data_dir  = get_parameter("episode_data_dir").as_string();
    const std::string traj_name = get_parameter("traj_name").as_string();
    const int    backtrack = get_parameter("backtrack").as_int();
    const double discount  = get_parameter("discount").as_double();

    std::string emb_path    = data_dir + "/" + traj_name + "/clip_embeddings.bin";
    std::string prompt_path = data_dir + "/" + traj_name + "/traj_prompt.txt";

    // --- load episode ---
    try {
      episode_ = std::make_unique<Episode>(
          Episode::load(emb_path, prompt_path, backtrack, discount));
      RCLCPP_INFO(get_logger(),
                  "Episode loaded: %d frames, %zu instructions",
                  episode_->size(),
                  episode_->instructions.size());
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Failed to load episode: %s", e.what());
      throw;
    }

    // reset particles to uniform distribution over episode
    pf_.reset(episode_.get());

    // --- pub/sub ---
    using Float32MA = std_msgs::msg::Float32MultiArray;
    using String    = std_msgs::msg::String;
    using Int32     = std_msgs::msg::Int32;

    feature_sub_ = create_subscription<Float32MA>(
        "/image_feature", 10,
        [this](const Float32MA::SharedPtr msg) { feature_callback(msg); });

    prompt_pub_   = create_publisher<String>("/prompt",         10);
    time_idx_pub_ = create_publisher<Int32>("/pfoe/time_idx",   10);

    RCLCPP_INFO(get_logger(), "pfoe_node ready");
  }

private:
  void feature_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
  {
    Event ev;
    ev.observation.assign(msg->data.begin(), msg->data.end());

    episode_->push_back(ev);
    if (episode_->size() < 2) return;

    pf_.update(episode_.get());
    int t_star = pf_.best_time_idx();

    // clamp index to valid instruction range
    int inst_idx = t_star - 1;
    if (inst_idx < 0) inst_idx = 0;
    if (inst_idx >= static_cast<int>(episode_->instructions.size()))
      inst_idx = static_cast<int>(episode_->instructions.size()) - 1;

    auto prompt_msg = std_msgs::msg::String();
    prompt_msg.data = episode_->instructions[inst_idx];
    prompt_pub_->publish(prompt_msg);

    auto idx_msg = std_msgs::msg::Int32();
    idx_msg.data = t_star;
    time_idx_pub_->publish(idx_msg);
  }

  std::unique_ptr<Episode>                                        episode_;
  ParticleFilter                                                  pf_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr feature_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr               prompt_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr                time_idx_pub_;
};

}  // namespace pfoe

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pfoe::PFoENode>());
  rclcpp::shutdown();
  return 0;
}
