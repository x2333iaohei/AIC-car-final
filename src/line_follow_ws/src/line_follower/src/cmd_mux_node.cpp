#include <memory>
#include <string>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>

class CmdMuxNode : public rclcpp::Node
{
public:
  CmdMuxNode()
  : Node("cmd_mux_node")
  {
    stop_topic_ =
      this->declare_parameter<std::string>("stop_topic", "/cmd_vel_stop");
    stop_timeout_ =
      this->declare_parameter<double>("stop_timeout", 0.5);

    avoid_topic_ =
      this->declare_parameter<std::string>("avoid_topic", "/cmd_vel_avoid");
    avoid_timeout_ =
      this->declare_parameter<double>("avoid_timeout", 0.12);

    high_topic_ =
      this->declare_parameter<std::string>("high_topic", "/cmd_vel_detection");
    high_timeout_ =
      this->declare_parameter<double>("high_timeout", 0.12);

    low_topic_ =
      this->declare_parameter<std::string>("low_topic", "/cmd_vel_follower");
    output_topic_ =
      this->declare_parameter<std::string>("output_topic", "/cmd_vel");

    const auto now = this->now();
    last_stop_time_ = now;
    last_avoid_time_ = now;
    last_high_time_ = now;

    stop_sub_ =
      this->create_subscription<geometry_msgs::msg::Twist>(
        stop_topic_,
        rclcpp::QoS(10),
        std::bind(&CmdMuxNode::stop_callback, this, std::placeholders::_1));

    avoid_sub_ =
      this->create_subscription<geometry_msgs::msg::Twist>(
        avoid_topic_,
        rclcpp::QoS(10),
        std::bind(&CmdMuxNode::avoid_callback, this, std::placeholders::_1));

    high_sub_ =
      this->create_subscription<geometry_msgs::msg::Twist>(
        high_topic_,
        rclcpp::QoS(10),
        std::bind(&CmdMuxNode::high_callback, this, std::placeholders::_1));

    low_sub_ =
      this->create_subscription<geometry_msgs::msg::Twist>(
        low_topic_,
        rclcpp::QoS(10),
        std::bind(&CmdMuxNode::low_callback, this, std::placeholders::_1));

    pub_ =
      this->create_publisher<geometry_msgs::msg::Twist>(output_topic_, 10);

    RCLCPP_INFO(
      get_logger(),
      "Cmd mux (4-level): stop='%s'(%.2fs) > avoid='%s'(%.2fs) > high='%s'(%.2fs) > low='%s' -> '%s'",
      stop_topic_.c_str(), stop_timeout_,
      avoid_topic_.c_str(), avoid_timeout_,
      high_topic_.c_str(), high_timeout_,
      low_topic_.c_str(), output_topic_.c_str());
  }

private:
  bool stop_active(const rclcpp::Time & now) const
  {
    return has_stop_ && (now - last_stop_time_).seconds() <= stop_timeout_;
  }

  bool avoid_active(const rclcpp::Time & now) const
  {
    return has_avoid_ && (now - last_avoid_time_).seconds() <= avoid_timeout_;
  }

  bool high_active(const rclcpp::Time & now) const
  {
    return has_high_ && (now - last_high_time_).seconds() <= high_timeout_;
  }

  // 第一优先级：黄线/永久停车
  void stop_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    last_stop_msg_ = *msg;
    last_stop_time_ = this->now();
    has_stop_ = true;
    publish(*msg);
  }

  // 第二优先级：近距高风险红锥/红色禁行区紧急避障
  void avoid_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    const auto now = this->now();
    last_avoid_msg_ = *msg;
    last_avoid_time_ = now;
    has_avoid_ = true;

    if (stop_active(now)) {
      publish(last_stop_msg_);
      return;
    }
    publish(*msg);
  }

  // 第三优先级：YOLO 分岔/加分数字转向
  void high_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    const auto now = this->now();
    last_high_msg_ = *msg;
    last_high_time_ = now;
    has_high_ = true;

    if (stop_active(now)) {
      publish(last_stop_msg_);
      return;
    }
    if (avoid_active(now)) {
      publish(last_avoid_msg_);
      return;
    }
    publish(*msg);
  }

  // 第四优先级：常规视觉巡线与丁字闭环并线
  void low_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    const auto now = this->now();
    if (stop_active(now)) {
      publish(last_stop_msg_);
      return;
    }
    if (avoid_active(now)) {
      publish(last_avoid_msg_);
      return;
    }
    if (high_active(now)) {
      publish(last_high_msg_);
      return;
    }
    publish(*msg);
  }

  void publish(const geometry_msgs::msg::Twist & msg)
  {
    if (pub_) {
      pub_->publish(msg);
    }
  }

  std::string stop_topic_;
  std::string avoid_topic_;
  std::string high_topic_;
  std::string low_topic_;
  std::string output_topic_;

  double stop_timeout_{0.5};
  double avoid_timeout_{0.12};
  double high_timeout_{0.12};

  rclcpp::Time last_stop_time_;
  rclcpp::Time last_avoid_time_;
  rclcpp::Time last_high_time_;

  bool has_stop_{false};
  bool has_avoid_{false};
  bool has_high_{false};

  geometry_msgs::msg::Twist last_stop_msg_;
  geometry_msgs::msg::Twist last_avoid_msg_;
  geometry_msgs::msg::Twist last_high_msg_;

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr stop_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr avoid_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr high_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr low_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CmdMuxNode>());
  rclcpp::shutdown();
  return 0;
}
