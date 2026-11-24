#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <ai_msgs/msg/perception_targets.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

namespace
{
constexpr double kDefaultConfThreshold = 0.38;
constexpr int kDefaultDetFrameWidth = 640;
constexpr int kDefaultDetFrameHeight = 480;
constexpr int kDefaultFrameWindow = 2;
constexpr double kDefaultTurnCooldown = 1.0;
constexpr double kDefaultTurnDuration = 0.44;
constexpr double kDefaultTurnLinear = 0.20;
constexpr double kDefaultYawGain = 1.65;
constexpr double kDefaultMaxYaw = 1.00;
constexpr double kDefaultTargetYOffset = 0.0;
constexpr double kDefaultYAlignTolerance = 0.012;
constexpr double kDefaultTurnDirectionSign = -1.0;
constexpr double kDefaultCentroidTimeout = 0.30;
constexpr double kMinValidDetY = 0.48;
constexpr double kMaxValidDetY = 0.72;
constexpr const char * kDefaultLineCentroidTopic = "line_follower/centroid";
constexpr const char * kDefaultStateTopic = "line_follower/track_state";
constexpr const char * kDefaultStartTopic = "line_follower/start_cmd";
}  // namespace

class DetectionTurnController : public rclcpp::Node
{
public:
  DetectionTurnController()
  : Node("detection_turn_controller")
  {
    detection_topic_ =
      this->declare_parameter<std::string>("detection_topic", "hobot_dnn_detection");
    state_topic_ =
      this->declare_parameter<std::string>("state_topic", kDefaultStateTopic);
    start_topic_ =
      this->declare_parameter<std::string>("start_topic", kDefaultStartTopic);
    cmd_vel_topic_ =
      this->declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel_detection");

    conf_threshold_ =
      this->declare_parameter<double>("confidence_threshold", kDefaultConfThreshold);
    det_frame_width_ =
      this->declare_parameter<int>("det_frame_width", kDefaultDetFrameWidth);
    det_frame_height_ =
      this->declare_parameter<int>("det_frame_height", kDefaultDetFrameHeight);
    frame_window_ = std::max(
      1, static_cast<int>(this->declare_parameter<int>("frame_window", kDefaultFrameWindow)));
    min_candidate_frames_ = std::max(
      1, static_cast<int>(this->declare_parameter<int>("min_candidate_frames", 1)));
    turn_cooldown_ =
      this->declare_parameter<double>("turn_cooldown", kDefaultTurnCooldown);
    turn_duration_ =
      this->declare_parameter<double>("turn_duration", kDefaultTurnDuration);
    turn_linear_ =
      this->declare_parameter<double>("turn_linear", kDefaultTurnLinear);
    yaw_gain_ =
      this->declare_parameter<double>("yaw_gain", kDefaultYawGain);
    max_yaw_rate_ =
      this->declare_parameter<double>("max_yaw_rate", kDefaultMaxYaw);
    target_y_offset_ =
      this->declare_parameter<double>("target_y_offset", kDefaultTargetYOffset);
    y_align_tolerance_ =
      this->declare_parameter<double>("y_align_tolerance", kDefaultYAlignTolerance);
    turn_direction_sign_ =
      this->declare_parameter<double>("turn_direction_sign", kDefaultTurnDirectionSign);

    line_centroid_topic_ =
      this->declare_parameter<std::string>("line_centroid_topic", kDefaultLineCentroidTopic);
    centroid_timeout_ =
      this->declare_parameter<double>("centroid_timeout", kDefaultCentroidTimeout);

    const auto now = this->now();
    turning_end_time_ = now;
    turn_cooldown_until_ = now;
    last_centroid_time_ = rclcpp::Time(0, 0, this->get_clock()->get_clock_type());

    detection_sub_ = this->create_subscription<ai_msgs::msg::PerceptionTargets>(
      detection_topic_, rclcpp::QoS(10).best_effort(),
      std::bind(&DetectionTurnController::detection_callback, this, std::placeholders::_1));

    centroid_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
      line_centroid_topic_, rclcpp::QoS(10),
      std::bind(&DetectionTurnController::centroid_callback, this, std::placeholders::_1));

    state_sub_ = this->create_subscription<std_msgs::msg::String>(
      state_topic_, rclcpp::QoS(10),
      std::bind(&DetectionTurnController::state_callback, this, std::placeholders::_1));

    start_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      start_topic_, rclcpp::QoS(10),
      std::bind(&DetectionTurnController::start_callback, this, std::placeholders::_1));

    cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);

    publish_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(50), std::bind(&DetectionTurnController::on_timer, this));

    RCLCPP_INFO(
      get_logger(),
      "Detection turn controller listening '%s', start '%s', centroid '%s', publishing '%s' "
      "(conf>=%.2f, y_off=%+.2f, y_tol=%.3f, win=%d, dur=%.2fs)",
      detection_topic_.c_str(), start_topic_.c_str(), line_centroid_topic_.c_str(),
      cmd_vel_topic_.c_str(), conf_threshold_, target_y_offset_, y_align_tolerance_,
      frame_window_, turn_duration_);
  }

private:
  struct Candidate
  {
    int digit{-1};
    double conf{0.0};
    double x_norm{0.5};
    double y_norm{0.5};
  };

  static int parse_digit_label(const std::string & label)
  {
    if (label.empty()) {
      return -1;
    }
    size_t start = (label[0] == 'd' || label[0] == 'D') ? 1 : 0;
    if (start >= label.size()) {
      return -1;
    }
    for (size_t i = start; i < label.size(); ++i) {
      if (!std::isdigit(static_cast<unsigned char>(label[i]))) {
        return -1;
      }
    }
    int value = -1;
    try {
      value = std::stoi(label.substr(start));
    } catch (...) {
      return -1;
    }
    return (value >= 1 && value <= 8) ? value : -1;
  }

  void clear_session()
  {
    session_active_ = false;
    session_frames_ = 0;
    best_digit_ = -1;
    best_conf_ = 0.0;
    best_x_diff_ = 0.0;
    best_y_diff_ = 0.0;
  }

  void start_callback(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    start_received_ = true;
    const bool new_state = msg->data;
    if (!new_state && start_enabled_) {
      clear_session();
      turning_ = false;
    }
    start_enabled_ = new_state;
  }

  void centroid_callback(const geometry_msgs::msg::Point::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    last_centroid_norm_x_ = std::clamp(static_cast<double>(msg->x), 0.0, 1.0);
    last_centroid_norm_y_ = std::clamp(static_cast<double>(msg->y), 0.0, 1.0);
    last_centroid_time_ = this->now();
    has_centroid_ = true;
  }

  void state_callback(const std_msgs::msg::String::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    const auto & s = msg->data;
    if (s == "yellow_line_detected") {
      yellow_stopped_ = true;
      clear_session();
      turning_ = false;
      turn_cooldown_until_ = this->now();
    } else {
      yellow_stopped_ = false;
    }
  }

  void detection_callback(const ai_msgs::msg::PerceptionTargets::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    if ((start_received_ && !start_enabled_) || yellow_stopped_) {
      return;
    }

    const rclcpp::Time now = this->now();
    if (turning_ || now < turn_cooldown_until_) {
      return;
    }

    if (!has_centroid_ || (now - last_centroid_time_).seconds() > centroid_timeout_) {
      return;
    }

    if (!session_active_) {
      clear_session();
      session_active_ = true;
    }

    const double full_w =
      det_frame_width_ > 0 ? static_cast<double>(det_frame_width_) : 640.0;
    const double full_h =
      det_frame_height_ > 0 ? static_cast<double>(det_frame_height_) : 480.0;

    // 关键设计：先在有效前视视野 [0.48, 0.72] 内找出全图最大面值数字 chosen，
    // 再检查该最大数字自身是否到达对齐线 target_y！
    // 这样当弯道外侧低分数字（如 2 或 5）比高分数字（7、8、4）提前 0.2~1.0s 经过对齐线时，
    // 绝不会被低分数字抢先触发冷却，而是精准等待全图最大数字到达对齐线再转向！
    std::vector<Candidate> candidates;
    for (const auto & target : msg->targets) {
      for (const auto & roi : target.rois) {
        if (roi.rect.width <= 0 || roi.rect.height <= 0 || roi.confidence < conf_threshold_) {
          continue;
        }
        const int digit = parse_digit_label(roi.type);
        if (digit < 1 || digit > 8) {
          continue;
        }
        const double cx_norm = std::clamp(
          (static_cast<double>(roi.rect.x_offset) + static_cast<double>(roi.rect.width) * 0.5) /
            full_w,
          0.0, 1.0);
        const double cy_norm = std::clamp(
          (static_cast<double>(roi.rect.y_offset) + static_cast<double>(roi.rect.height) * 0.5) /
            full_h,
          0.0, 1.0);
        if (cy_norm < kMinValidDetY || cy_norm > kMaxValidDetY) {
          continue;
        }
        candidates.push_back(
          Candidate{digit, static_cast<double>(roi.confidence), cx_norm, cy_norm});
      }
    }

    if (!candidates.empty()) {
      const auto chosen = *std::max_element(
        candidates.begin(), candidates.end(),
        [](const Candidate & a, const Candidate & b) {
          if (a.digit != b.digit) {
            return a.digit < b.digit;
          }
          return a.conf < b.conf;
        });

      const double target_y =
        std::clamp(last_centroid_norm_y_ + target_y_offset_, 0.0, 1.0);
      const double y_diff = chosen.y_norm - target_y;
      if (std::abs(y_diff) <= y_align_tolerance_) {
        const double x_diff = chosen.x_norm - last_centroid_norm_x_;
        if (chosen.digit > best_digit_ ||
            (chosen.digit == best_digit_ && chosen.conf > best_conf_))
        {
          best_digit_ = chosen.digit;
          best_conf_ = chosen.conf;
          best_x_diff_ = x_diff;
          best_y_diff_ = y_diff;
        }
      }
    }

    session_frames_++;
    if (session_frames_ >= frame_window_) {
      trigger_turn(now);
      clear_session();
    }
  }

  void trigger_turn(const rclcpp::Time & now)
  {
    if (best_digit_ < 0) {
      return;
    }

    const double yaw = std::clamp(
      turn_direction_sign_ * best_x_diff_ * yaw_gain_, -max_yaw_rate_, max_yaw_rate_);
    if (std::abs(yaw) < 1e-4) {
      return;
    }

    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = turn_linear_;
    cmd.angular.z = yaw;

    turning_cmd_ = cmd;
    turning_end_time_ = now + rclcpp::Duration::from_seconds(turn_duration_);
    turn_cooldown_until_ = turning_end_time_ + rclcpp::Duration::from_seconds(turn_cooldown_);
    turning_ = true;

    cmd_pub_->publish(turning_cmd_);

    RCLCPP_INFO(
      get_logger(),
      "[digit turn] best:%d conf:%.2f x_diff:%.2f y_diff:%.2f dur:%.2fs lin:%.2f ang:%.2f (win:%d)",
      best_digit_, best_conf_, best_x_diff_, best_y_diff_,
      turn_duration_, cmd.linear.x, cmd.angular.z, session_frames_);
  }

  void on_timer()
  {
    if (!turning_) {
      return;
    }
    if ((start_received_ && !start_enabled_) || yellow_stopped_ || this->now() >= turning_end_time_) {
      turning_ = false;
      return;
    }
    cmd_pub_->publish(turning_cmd_);
  }

  std::string detection_topic_;
  std::string state_topic_;
  std::string start_topic_;
  std::string cmd_vel_topic_;
  std::string line_centroid_topic_;
  double conf_threshold_{kDefaultConfThreshold};
  int det_frame_width_{kDefaultDetFrameWidth};
  int det_frame_height_{kDefaultDetFrameHeight};
  int frame_window_{kDefaultFrameWindow};
  int min_candidate_frames_{1};
  double turn_cooldown_{kDefaultTurnCooldown};
  double turn_duration_{kDefaultTurnDuration};
  double turn_linear_{kDefaultTurnLinear};
  double yaw_gain_{kDefaultYawGain};
  double max_yaw_rate_{kDefaultMaxYaw};
  double target_y_offset_{kDefaultTargetYOffset};
  double y_align_tolerance_{kDefaultYAlignTolerance};
  double turn_direction_sign_{kDefaultTurnDirectionSign};
  double centroid_timeout_{kDefaultCentroidTimeout};

  bool start_received_{false};
  bool start_enabled_{false};
  bool yellow_stopped_{false};
  bool has_centroid_{false};

  bool turning_{false};
  geometry_msgs::msg::Twist turning_cmd_;
  rclcpp::Time turning_end_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time turn_cooldown_until_{0, 0, RCL_ROS_TIME};

  bool session_active_{false};
  int session_frames_{0};
  int best_digit_{-1};
  double best_conf_{0.0};
  double best_x_diff_{0.0};
  double best_y_diff_{0.0};

  double last_centroid_norm_x_{0.5};
  double last_centroid_norm_y_{0.5};
  rclcpp::Time last_centroid_time_{0, 0, RCL_ROS_TIME};

  rclcpp::Subscription<ai_msgs::msg::PerceptionTargets>::SharedPtr detection_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr centroid_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr start_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DetectionTurnController>());
  rclcpp::shutdown();
  return 0;
}
