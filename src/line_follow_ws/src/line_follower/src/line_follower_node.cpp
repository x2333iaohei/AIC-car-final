#include "line_follower/line_follower_node.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <utility>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/point.hpp>
#include <opencv2/imgproc.hpp>
#include <std_msgs/msg/bool.hpp>

namespace
{

constexpr double kPi = 3.14159265358979323846;

struct CurveAssistRuntime
{
  bool reverse_curve_active{false};
  int reverse_confirm_count{0};
  int reverse_clear_count{0};
  bool near_valid{false};
  bool far_valid{false};
  double near_error{0.0};
  double far_error{0.0};
  double curve_level{0.0};
  double far_weight{0.30};
};

CurveAssistRuntime g_curve_assist;

struct RedAvoidRuntime
{
  bool valid{false};

  // 横向红色停车线或镂空红色禁行区聚类框：与实心红锥分类处理，均只绕行不停车
  bool red_line_detected{false};
  bool is_red_line{false};
  cv::Rect red_line_bbox_img{};

  cv::Rect bbox_img{};
  cv::Point2d center_img{};
  cv::Point2d bottom_img{};

  double aspect_ratio{0.0};
  double contour_area_ratio{0.0};
  double bottom_ratio{0.0};

  bool in_corridor{false};
  double risk{0.0};
  bool very_close{false};
};

RedAvoidRuntime g_red_avoid;

struct MotionRuntime
{
  bool valid{false};
  double last_linear{0.08};
  double last_yaw{0.0};
};

MotionRuntime g_motion;

struct BandRun
{
  double center{0.0};
  int width{0};
};

}  // namespace

namespace line_follower
{

LineFollowerNode::LineFollowerNode()
: Node("line_follower_node")
{
  declare_parameters();
  setup_interfaces();

  const int kernel_size = std::max<int>(1, morph_kernel_);
  morph_element_ = cv::getStructuringElement(
    cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));

  const auto now = this->now();
  last_detection_time_ = now;
  yellow_line_detection_time_ = now;
  last_error_time_ = now;
  intersection_end_time_ = now;
  intersection_cooldown_until_ = now;
  tjunction_merge_start_time_ = now;
  red_turn_end_time_ = now;

  last_state_ = TrackState::LOST;
  yellow_line_stop_active_ = false;
  yellow_line_detection_count_ = 0;
  yellow_line_latched_ = false;
  yellow_clear_frames_ = 0;
  permanent_stop_ = false;
  start_enabled_ = false;
  red_left_votes_ = 0;
  red_right_votes_ = 0;
  red_turning_ = false;
  last_red_overlay_valid_ = false;

  RCLCPP_INFO(
    this->get_logger(),
    "Line follower subscribing '%s', cmd '%s', avoid '%s', det_cmd '%s', start '%s' "
    "(kp=%.2f, kd=%.2f, preview=%s, tjunction=%s, red_zone_cluster=%s)",
    image_topic_.c_str(), cmd_vel_topic_.c_str(), avoid_cmd_topic_.c_str(),
    detection_cmd_topic_.c_str(), start_topic_.c_str(), kp_, kd_,
    this->get_parameter("enable_curve_preview_steering").as_bool() ? "on" : "off",
    this->get_parameter("enable_tjunction_merge").as_bool() ? "on" : "off",
    this->get_parameter("enable_red_zone_cluster").as_bool() ? "on" : "off");
}

void LineFollowerNode::declare_parameters()
{
  this->declare_parameter("image_topic", "/ascamera/camera_publisher/rgb/image0");
  this->declare_parameter("cmd_vel_topic", "/cmd_vel");
  this->declare_parameter("avoid_cmd_topic", "/cmd_vel_avoid");
  this->declare_parameter("detection_cmd_topic", "/cmd_vel_detection");
  this->declare_parameter("debug_image_topic", "line_follower/debug_image");
  this->declare_parameter("line_state_topic", "line_follower/track_state");
  this->declare_parameter("start_topic", "line_follower/start_cmd");
  this->declare_parameter("stop_cmd_topic", "/cmd_vel_stop");
  this->declare_parameter("line_centroid_topic", "line_follower/centroid");

  this->declare_parameter("target_speed", 0.6);
  this->declare_parameter("max_speed", 1.0);
  this->declare_parameter("min_speed", 0.0);
  this->declare_parameter("max_yaw_rate", 1.2);
  this->declare_parameter("kp", 1.8);
  this->declare_parameter("kd", 0.0);

  this->declare_parameter("roi_top_ratio", 0.5);
  this->declare_parameter("roi_bottom_ratio", 0.05);
  this->declare_parameter("gaussian_kernel", 5);
  this->declare_parameter("binary_threshold", 160.0);
  this->declare_parameter("morph_kernel", 5);
  this->declare_parameter("morph_iterations", 2);
  this->declare_parameter("min_contour_area", 2000.0);
  this->declare_parameter("invert_binary", false);
  this->declare_parameter("use_white_filter", false);
  this->declare_parameter("white_h_low", 0);
  this->declare_parameter("white_h_high", 180);
  this->declare_parameter("white_s_max", 60);
  this->declare_parameter("white_v_min", 180);

  this->declare_parameter("use_yellow_filter", false);
  this->declare_parameter("yellow_h_low", 20);
  this->declare_parameter("yellow_h_high", 40);
  this->declare_parameter("yellow_s_min", 80);
  this->declare_parameter("yellow_v_min", 120);
  this->declare_parameter("yellow_area_ratio_threshold", 0.1);
  this->declare_parameter("yellow_stop_duration", 1.0);
  this->declare_parameter("yellow_permanent_count", 5);
  this->declare_parameter("yellow_clear_frames", 10);

  this->declare_parameter("enable_intersection_probe", false);
  this->declare_parameter("intersection_band_height", 50);
  this->declare_parameter("intersection_dark_ratio_threshold", 0.9);
  this->declare_parameter("intersection_turn_speed", 0.2);
  this->declare_parameter("intersection_turn_yaw_rate", 0.8);
  this->declare_parameter("intersection_turn_duration", 1.0);
  this->declare_parameter("intersection_cooldown", 2.0);

  // 红色避障
  this->declare_parameter("enable_red_filter", false);
  this->declare_parameter("red_h_low_1", 0);
  this->declare_parameter("red_h_high_1", 10);
  this->declare_parameter("red_h_low_2", 160);
  this->declare_parameter("red_h_high_2", 180);
  this->declare_parameter("red_s_min", 100);
  this->declare_parameter("red_v_min", 80);
  this->declare_parameter("red_area_ratio_threshold", 0.005);
  this->declare_parameter("red_turn_yaw", 0.6);
  this->declare_parameter("red_turn_linear", 0.0);
  this->declare_parameter("red_turn_duration", 1.0);
  this->declare_parameter("red_vote_frames", 8);
  this->declare_parameter("red_divider_angle_deg", 0.0);

  // 连续弯辅助与动态远端预瞄（默认关闭，完全向后兼容）
  this->declare_parameter("enable_continuous_curve_assist", false);
  this->declare_parameter("reverse_curve_min_error", 0.12);
  this->declare_parameter("reverse_curve_threshold", 0.35);
  this->declare_parameter("reverse_curve_confirm_frames", 3);
  this->declare_parameter("reverse_curve_clear_frames", 4);
  this->declare_parameter("reverse_curve_speed", 0.16);
  this->declare_parameter("sharp_curve_threshold", 0.50);
  this->declare_parameter("sharp_curve_speed", 0.20);
  this->declare_parameter("enable_curve_preview_steering", false);
  this->declare_parameter("curve_preview_max_far_weight", 0.45);

  // `┤` 丁字汇合口 DAG 顺行闭环并线（默认关闭，仅岔路图启用）
  this->declare_parameter("enable_tjunction_merge", false);
  this->declare_parameter("tjunction_require_branch_gate", true);
  this->declare_parameter("tjunction_branch_yellow_count", 2);
  this->declare_parameter("tjunction_branch_trigger_sign", -1);
  this->declare_parameter("tjunction_merge_direction", -1);
  this->declare_parameter("tjunction_confirm_frames", 3);
  this->declare_parameter("tjunction_wall_min_y_ratio", 0.23);
  this->declare_parameter("tjunction_wall_max_y_ratio", 0.62);
  this->declare_parameter("tjunction_wall_max_spread_ratio", 0.10);
  this->declare_parameter("tjunction_min_full_rows", 55);
  this->declare_parameter("tjunction_merge_speed", 0.16);
  this->declare_parameter("tjunction_merge_yaw_rate", 1.15);
  this->declare_parameter("tjunction_min_turn_duration", 0.35);
  this->declare_parameter("tjunction_max_turn_duration", 1.25);
  this->declare_parameter("tjunction_exit_upper_road_ratio", 0.12);

  // 柔性红锥与镂空红色禁行区避让
  this->declare_parameter("enable_soft_red_avoidance", false);
  this->declare_parameter("red_soft_yaw", 0.30);
  this->declare_parameter("red_near_ratio", 0.35);
  this->declare_parameter("red_very_close_ratio", 0.72);
  this->declare_parameter("red_corridor_half_width_ratio", 0.42);
  this->declare_parameter("red_soft_speed", 0.15);
  this->declare_parameter("red_soft_curve_yaw_limit", 0.15);

  this->declare_parameter("red_line_aspect_ratio_min", 3.0);
  this->declare_parameter("red_line_min_width_ratio", 0.30);

  // 镂空红色禁行区（1cm 细框 + 多竖纹）闭运算聚类检测分支
  this->declare_parameter("enable_red_zone_cluster", false);
  this->declare_parameter("red_zone_close_kernel_w", 35);
  this->declare_parameter("red_zone_close_kernel_h", 13);
  this->declare_parameter("red_zone_min_width_ratio", 0.11);
  this->declare_parameter("red_zone_min_area_ratio", 0.0030);
  this->declare_parameter("red_zone_max_fill_ratio", 0.30);
  this->declare_parameter("red_zone_near_ratio", 0.28);

  this->declare_parameter("red_cone_area_ratio_threshold", 0.0006);
  this->declare_parameter("red_cone_min_bbox_width", 4);
  this->declare_parameter("red_cone_min_bbox_height", 4);

  this->declare_parameter("red_repulsion_gain", 1.5);
  this->declare_parameter("red_lane_keep_min_factor", 0.40);
  this->declare_parameter("red_min_forward_speed", 0.06);
  this->declare_parameter("red_mux_preempt_risk", 0.28);

  this->declare_parameter("red_line_fallback_direction", -1);
  this->declare_parameter("red_line_soft_yaw", 0.35);
  this->declare_parameter("red_line_speed", 0.12);

  this->declare_parameter("line_lost_stop_time", 0.8);
  this->declare_parameter("line_lost_forward_speed", 0.06);
  this->declare_parameter("line_lost_yaw_hold_factor", 0.85);
  this->declare_parameter("speed_gain_damping", 0.5);
  this->declare_parameter("publish_debug_image", true);

  image_topic_ = this->get_parameter("image_topic").as_string();
  cmd_vel_topic_ = this->get_parameter("cmd_vel_topic").as_string();
  avoid_cmd_topic_ = this->get_parameter("avoid_cmd_topic").as_string();
  detection_cmd_topic_ = this->get_parameter("detection_cmd_topic").as_string();
  debug_image_topic_ = this->get_parameter("debug_image_topic").as_string();
  line_state_topic_ = this->get_parameter("line_state_topic").as_string();
  start_topic_ = this->get_parameter("start_topic").as_string();
  stop_cmd_topic_ = this->get_parameter("stop_cmd_topic").as_string();
  line_centroid_topic_ = this->get_parameter("line_centroid_topic").as_string();

  target_speed_ = this->get_parameter("target_speed").as_double();
  max_speed_ = this->get_parameter("max_speed").as_double();
  min_speed_ = this->get_parameter("min_speed").as_double();
  max_yaw_rate_ = this->get_parameter("max_yaw_rate").as_double();
  kp_ = this->get_parameter("kp").as_double();
  kd_ = std::max(0.0, this->get_parameter("kd").as_double());

  roi_top_ratio_ = this->get_parameter("roi_top_ratio").as_double();
  roi_bottom_ratio_ = this->get_parameter("roi_bottom_ratio").as_double();

  gaussian_kernel_ = this->get_parameter("gaussian_kernel").as_int();
  if (gaussian_kernel_ <= 0) {
    gaussian_kernel_ = 1;
  }
  if (gaussian_kernel_ % 2 == 0) {
    ++gaussian_kernel_;
  }

  binary_threshold_ = this->get_parameter("binary_threshold").as_double();
  morph_kernel_ =
    std::max<int>(1, static_cast<int>(this->get_parameter("morph_kernel").as_int()));
  morph_iterations_ =
    std::max<int>(1, static_cast<int>(this->get_parameter("morph_iterations").as_int()));
  min_contour_area_ = this->get_parameter("min_contour_area").as_double();

  invert_binary_ = this->get_parameter("invert_binary").as_bool();
  use_white_filter_ = this->get_parameter("use_white_filter").as_bool();
  white_h_low_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("white_h_low").as_int()), 0, 180);
  white_h_high_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("white_h_high").as_int()), 0, 180);
  if (white_h_low_ > white_h_high_) {
    std::swap(white_h_low_, white_h_high_);
  }
  white_s_max_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("white_s_max").as_int()), 0, 255);
  white_v_min_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("white_v_min").as_int()), 0, 255);

  use_yellow_filter_ = this->get_parameter("use_yellow_filter").as_bool();
  yellow_h_low_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("yellow_h_low").as_int()), 0, 180);
  yellow_h_high_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("yellow_h_high").as_int()), 0, 180);
  if (yellow_h_low_ > yellow_h_high_) {
    std::swap(yellow_h_low_, yellow_h_high_);
  }
  yellow_s_min_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("yellow_s_min").as_int()), 0, 255);
  yellow_v_min_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("yellow_v_min").as_int()), 0, 255);
  yellow_area_ratio_threshold_ = this->get_parameter("yellow_area_ratio_threshold").as_double();
  yellow_stop_duration_ = this->get_parameter("yellow_stop_duration").as_double();
  yellow_permanent_count_ = std::max<int>(
    1, static_cast<int>(this->get_parameter("yellow_permanent_count").as_int()));
  yellow_clear_frames_required_ = std::max<int>(
    1, static_cast<int>(this->get_parameter("yellow_clear_frames").as_int()));

  enable_intersection_probe_ = this->get_parameter("enable_intersection_probe").as_bool();
  intersection_band_height_ = std::max<int>(
    1, static_cast<int>(this->get_parameter("intersection_band_height").as_int()));
  intersection_dark_ratio_threshold_ =
    this->get_parameter("intersection_dark_ratio_threshold").as_double();
  intersection_turn_speed_ = this->get_parameter("intersection_turn_speed").as_double();
  intersection_turn_yaw_rate_ = this->get_parameter("intersection_turn_yaw_rate").as_double();
  intersection_turn_duration_ = this->get_parameter("intersection_turn_duration").as_double();
  intersection_cooldown_ = this->get_parameter("intersection_cooldown").as_double();

  enable_red_filter_ = this->get_parameter("enable_red_filter").as_bool();
  red_h_low_1_ = this->get_parameter("red_h_low_1").as_int();
  red_h_high_1_ = this->get_parameter("red_h_high_1").as_int();
  red_h_low_2_ = this->get_parameter("red_h_low_2").as_int();
  red_h_high_2_ = this->get_parameter("red_h_high_2").as_int();
  red_s_min_ = this->get_parameter("red_s_min").as_int();
  red_v_min_ = this->get_parameter("red_v_min").as_int();
  red_area_ratio_threshold_ = this->get_parameter("red_area_ratio_threshold").as_double();
  red_turn_yaw_ = this->get_parameter("red_turn_yaw").as_double();
  red_turn_linear_ = this->get_parameter("red_turn_linear").as_double();
  red_turn_duration_ = this->get_parameter("red_turn_duration").as_double();
  red_vote_frames_ = std::max<int>(1, this->get_parameter("red_vote_frames").as_int());
  red_divider_angle_deg_ = this->get_parameter("red_divider_angle_deg").as_double();
  red_divider_angle_rad_ = red_divider_angle_deg_ * (kPi / 180.0);

  line_lost_stop_time_ = this->get_parameter("line_lost_stop_time").as_double();
  speed_gain_damping_ = this->get_parameter("speed_gain_damping").as_double();
  publish_debug_image_ = this->get_parameter("publish_debug_image").as_bool();
}

void LineFollowerNode::setup_interfaces()
{
  cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);
  if (!avoid_cmd_topic_.empty()) {
    avoid_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(avoid_cmd_topic_, 10);
  }
  stop_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(stop_cmd_topic_, 10);
  state_pub_ = this->create_publisher<std_msgs::msg::String>(line_state_topic_, 10);
  centroid_pub_ = this->create_publisher<geometry_msgs::msg::Point>(line_centroid_topic_, 10);

  image_sub_ = image_transport::create_subscription(
    this,
    image_topic_,
    std::bind(&LineFollowerNode::image_callback, this, std::placeholders::_1),
    "raw");

  start_sub_ = this->create_subscription<std_msgs::msg::Bool>(
    start_topic_, 10, std::bind(&LineFollowerNode::start_callback, this, std::placeholders::_1));

  if (!detection_cmd_topic_.empty()) {
    detection_cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      detection_cmd_topic_, 10,
      std::bind(&LineFollowerNode::detection_cmd_callback, this, std::placeholders::_1));
  }

  if (publish_debug_image_) {
    debug_pub_ = image_transport::create_publisher(this, debug_image_topic_);
    debug_enabled_ = true;
  }
}

void LineFollowerNode::start_callback(const std_msgs::msg::Bool::SharedPtr msg)
{
  const bool new_state = msg->data;
  if (new_state == start_enabled_) {
    return;
  }

  const bool old_state = start_enabled_;

  if (!old_state && new_state && permanent_stop_) {
    const auto now = this->now();
    yellow_line_detection_count_ = 0;
    yellow_line_stop_active_ = false;
    yellow_line_latched_ = false;
    yellow_clear_frames_ = 0;
    permanent_stop_ = false;
    yellow_line_detection_time_ = now;

    red_left_votes_ = 0;
    red_right_votes_ = 0;
    red_turning_ = false;
    red_turn_end_time_ = now;

    intersection_active_ = false;
    intersection_end_time_ = now;
    intersection_cooldown_until_ = now;

    tjunction_branch_decided_ = false;
    tjunction_merge_armed_ = false;
    tjunction_confirm_count_ = 0;
    tjunction_merging_ = false;
    tjunction_merge_start_time_ = now;

    has_prev_error_ = false;
    filtered_deriv_ = 0.0;
    last_detection_time_ = now;
    last_state_ = TrackState::LOST;

    RCLCPP_INFO(
      this->get_logger(),
      "Previous lap finished. New lap reset: yellow count=0, permanent stop & T-merge cleared.");
  }

  start_enabled_ = new_state;

  if (!start_enabled_) {
    g_curve_assist = CurveAssistRuntime{};
    has_prev_error_ = false;
    filtered_deriv_ = 0.0;
    tjunction_confirm_count_ = 0;
    tjunction_merging_ = false;
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Start signal changed: %s -> %s",
    old_state ? "true" : "false",
    start_enabled_ ? "true" : "false");
}

void LineFollowerNode::detection_cmd_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
  if (!msg || !start_enabled_) {
    return;
  }
  if (!this->get_parameter("enable_tjunction_merge").as_bool()) {
    return;
  }

  const int branch_yellow_count =
    static_cast<int>(this->get_parameter("tjunction_branch_yellow_count").as_int());
  if (yellow_line_detection_count_ != branch_yellow_count || tjunction_branch_decided_) {
    return;
  }
  if (std::abs(msg->angular.z) < 0.05) {
    return;
  }

  tjunction_branch_decided_ = true;
  const int trigger_sign =
    static_cast<int>(this->get_parameter("tjunction_branch_trigger_sign").as_int());
  const bool entered_inner_branch =
    (trigger_sign < 0 && msg->angular.z < -0.05) ||
    (trigger_sign > 0 && msg->angular.z > 0.05);

  if (entered_inner_branch) {
    tjunction_merge_armed_ = true;
    tjunction_confirm_count_ = 0;
    RCLCPP_INFO(
      this->get_logger(),
      "[T-merge] Inner shortcut branch entered (det ang=%.2f at yellow=%d). T-junction merge armed.",
      msg->angular.z, yellow_line_detection_count_);
  } else {
    tjunction_merge_armed_ = false;
    RCLCPP_INFO(
      this->get_logger(),
      "[T-merge] Outer loop branch entered (det ang=%.2f at yellow=%d). T-junction merge disarmed.",
      msg->angular.z, yellow_line_detection_count_);
  }
}

void LineFollowerNode::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  cv::Mat frame_bgr;
  try {
    frame_bgr = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  auto result_opt = process_image(frame_bgr);

  if (result_opt && result_opt->line_detected) {
    double norm_x = 0.5;
    double norm_y = 0.5;
    if (frame_bgr.cols > 0 && frame_bgr.rows > 0) {
      norm_x = result_opt->centroid.x / static_cast<double>(frame_bgr.cols);
      norm_y = (static_cast<double>(result_opt->roi_top) + result_opt->centroid.y) /
        static_cast<double>(frame_bgr.rows);
      norm_x = clamp(norm_x, 0.0, 1.0);
      norm_y = clamp(norm_y, 0.0, 1.0);
    }
    publish_centroid(norm_x, norm_y);
  }

  if (!start_enabled_) {
    g_curve_assist = CurveAssistRuntime{};
    has_prev_error_ = false;
    filtered_deriv_ = 0.0;

    static rclcpp::Time last_log_time = this->now();
    rclcpp::Time current_time = this->now();
    if ((current_time - last_log_time).seconds() > 2.0) {
      RCLCPP_INFO(this->get_logger(), "Line follower is not started, waiting for start signal...");
      last_log_time = current_time;
    }
    if (result_opt && debug_enabled_) {
      publish_debug_image(msg, frame_bgr, *result_opt);
    }

    publish_cmd_vel(0.0, 0.0, true, false);
    return;
  }

  const auto now = this->now();

  double linear_speed = 0.0;
  double yaw_rate = 0.0;
  TrackState current_state = TrackState::LOST;
  bool publish_stop_cmd = false;

  // ============================================================
  // 红色避障检测与投票（支持实心红锥 + 镂空红色禁行区聚类）
  // ============================================================
  const bool soft_red_enabled =
    this->get_parameter("enable_soft_red_avoidance").as_bool();
  const double red_soft_yaw =
    std::max(0.0, this->get_parameter("red_soft_yaw").as_double());
  const double red_near_ratio = clamp(
    this->get_parameter("red_near_ratio").as_double(), 0.0, 1.0);
  const double red_zone_near_ratio = clamp(
    this->get_parameter("red_zone_near_ratio").as_double(), 0.0, 1.0);
  const double red_very_close_ratio = clamp(
    this->get_parameter("red_very_close_ratio").as_double(), 0.0, 1.0);
  const double red_corridor_half_width_ratio = clamp(
    this->get_parameter("red_corridor_half_width_ratio").as_double(), 0.0, 1.0);
  const double red_soft_speed =
    std::max(0.0, this->get_parameter("red_soft_speed").as_double());
  const double red_min_forward_speed =
    std::max(0.01, this->get_parameter("red_min_forward_speed").as_double());
  const double red_mux_preempt_risk = clamp(
    this->get_parameter("red_mux_preempt_risk").as_double(), 0.0, 1.0);

  const double line_lost_forward_speed =
    std::max(0.01, this->get_parameter("line_lost_forward_speed").as_double());
  const double line_lost_yaw_hold_factor =
    clamp(this->get_parameter("line_lost_yaw_hold_factor").as_double(), 0.0, 1.0);

  bool soft_red_active = false;
  double soft_red_yaw_bias = 0.0;
  double active_red_near_ratio = red_near_ratio;

  g_red_avoid.in_corridor = false;
  g_red_avoid.risk = 0.0;
  g_red_avoid.very_close = false;

  if (enable_red_filter_ && result_opt && result_opt->line_detected) {
    int roi_top = result_opt->roi_top;
    int roi_bottom = result_opt->roi_bottom;
    cv::Point2d line_centroid_roi(result_opt->centroid.x, result_opt->centroid.y);

    double side_sign = 0.0;
    cv::Point2d red_center_img, div_base_img, div_target_img;
    bool red_seen = detect_red(
      frame_bgr, roi_top, roi_bottom, result_opt->binary_mask, line_centroid_roi, side_sign,
      red_center_img, div_base_img, div_target_img);

    last_red_overlay_valid_ = red_seen;

    if (red_seen) {
      last_red_center_img_ = red_center_img;
      last_div_base_img_ = div_base_img;
      last_div_target_img_ = div_target_img;

      if (side_sign < 0.0) {
        red_left_votes_++;
        red_right_votes_ = 0;
      } else {
        red_right_votes_++;
        red_left_votes_ = 0;
      }

      if (soft_red_enabled) {
        red_turning_ = false;
        active_red_near_ratio =
          g_red_avoid.is_red_line ? red_zone_near_ratio : red_near_ratio;

        const double red_bottom_ratio = clamp(g_red_avoid.bottom_ratio, 0.0, 1.0);
        const bool vote_confirmed =
          red_left_votes_ >= red_vote_frames_ ||
          red_right_votes_ >= red_vote_frames_;

        const cv::Point2d corridor_v = div_target_img - div_base_img;
        const cv::Point2d red_v = g_red_avoid.bottom_img - div_base_img;
        const double corridor_norm = std::hypot(corridor_v.x, corridor_v.y);

        double lateral_distance = 1e9;
        if (corridor_norm > 1e-6) {
          lateral_distance = std::abs(
            corridor_v.x * red_v.y - corridor_v.y * red_v.x) / corridor_norm;
        }

        const double corridor_limit =
          red_corridor_half_width_ratio * static_cast<double>(frame_bgr.cols);
        const bool red_in_corridor = lateral_distance <= corridor_limit;
        g_red_avoid.in_corridor = red_in_corridor;

        const bool red_close = red_bottom_ratio >= active_red_near_ratio;
        const bool red_very_close = red_bottom_ratio >= red_very_close_ratio;

        if (vote_confirmed && red_close && (red_in_corridor || red_very_close)) {
          soft_red_active = true;
          const double base_avoid_yaw =
            g_red_avoid.is_red_line ?
            std::max(0.0, this->get_parameter("red_line_soft_yaw").as_double()) :
            red_soft_yaw;
          soft_red_yaw_bias = (side_sign < 0.0) ? -base_avoid_yaw : base_avoid_yaw;
        }
      } else {
        if (red_left_votes_ >= red_vote_frames_) {
          red_turn_yaw_cmd_ = clamp(-red_turn_yaw_, -max_yaw_rate_, max_yaw_rate_);
          red_turning_ = true;
          red_turn_end_time_ = now + rclcpp::Duration::from_seconds(red_turn_duration_);
          red_left_votes_ = 0;
          red_right_votes_ = 0;
        } else if (red_right_votes_ >= red_vote_frames_) {
          red_turn_yaw_cmd_ = clamp(red_turn_yaw_, -max_yaw_rate_, max_yaw_rate_);
          red_turning_ = true;
          red_turn_end_time_ = now + rclcpp::Duration::from_seconds(red_turn_duration_);
          red_left_votes_ = 0;
          red_right_votes_ = 0;
        }
      }
    } else {
      red_left_votes_ = 0;
      red_right_votes_ = 0;
      if (soft_red_enabled) {
        red_turning_ = false;
      } else if (now >= red_turn_end_time_) {
        red_turning_ = false;
      }
    }
  } else {
    last_red_overlay_valid_ = false;
    g_red_avoid.valid = false;
    g_red_avoid.red_line_detected = false;
    g_red_avoid.is_red_line = false;

    if (soft_red_enabled) {
      red_left_votes_ = 0;
      red_right_votes_ = 0;
      red_turning_ = false;
    }
  }

  // ============================================================
  // 黄色线计数（保持原实车稳定逻辑完全不变）
  // ============================================================
  const bool yellow_detected =
    result_opt && result_opt->yellow_line_detected;

  if (yellow_detected) {
    yellow_clear_frames_ = 0;

    if (!yellow_line_latched_) {
      yellow_line_latched_ = true;
      yellow_line_detection_count_++;
      yellow_line_detection_time_ = now;

      // 进入新一段拓扑区间：更新丁字并线门控状态
      tjunction_branch_decided_ = false;
      tjunction_confirm_count_ = 0;
      tjunction_merging_ = false;
      const int branch_yellow_count =
        static_cast<int>(this->get_parameter("tjunction_branch_yellow_count").as_int());
      if (yellow_line_detection_count_ != branch_yellow_count) {
        tjunction_merge_armed_ = false;
      }

      RCLCPP_INFO(
        this->get_logger(),
        "New yellow line detected! Count: %d/%d",
        yellow_line_detection_count_,
        yellow_permanent_count_);

      if (yellow_line_detection_count_ >= yellow_permanent_count_) {
        permanent_stop_ = true;
        yellow_line_stop_active_ = false;
        RCLCPP_INFO(
          this->get_logger(),
          "Final yellow line detected. Permanent stop activated.");
      } else {
        yellow_line_stop_active_ = true;
        RCLCPP_INFO(
          this->get_logger(),
          "Temporary yellow stop activated for %.2f s.",
          yellow_stop_duration_);
      }
    }
  } else if (yellow_line_latched_) {
    yellow_clear_frames_++;
    if (yellow_clear_frames_ >= yellow_clear_frames_required_) {
      yellow_line_latched_ = false;
      yellow_clear_frames_ = 0;
      RCLCPP_INFO(
        this->get_logger(),
        "Yellow line cleared. Ready for next yellow line.");
    }
  }

  const double since_yellow_detection =
    (now - yellow_line_detection_time_).seconds();

  // ============================================================
  // 遗留交叉口探针（默认关闭，保留兼容）
  // ============================================================
  if (enable_intersection_probe_ && result_opt && result_opt->line_detected &&
    !intersection_active_ && now >= intersection_cooldown_until_ &&
    !yellow_line_stop_active_ && !permanent_stop_)
  {
    const cv::Mat & mask = result_opt->binary_mask;
    if (!mask.empty()) {
      int band_h = std::min(intersection_band_height_, mask.rows);
      if (band_h > 0) {
        cv::Rect band_rect(0, 0, mask.cols, band_h);
        cv::Mat band = mask(band_rect);
        int nonzero = cv::countNonZero(band);
        double total = static_cast<double>(band.total());
        double black_ratio =
          total > 0.0 ? 1.0 - static_cast<double>(nonzero) / total : 0.0;

        if (black_ratio >= intersection_dark_ratio_threshold_) {
          intersection_active_ = true;
          intersection_end_time_ =
            now + rclcpp::Duration::from_seconds(intersection_turn_duration_);
          intersection_cooldown_until_ =
            now + rclcpp::Duration::from_seconds(intersection_cooldown_);
        }
      }
    }
  }

  // ============================================================
  // 基础控制：黄色停车 > PD 巡线 > 丢线低速继续
  // ============================================================
  if (permanent_stop_) {
    current_state = TrackState::YELLOW_LINE_DETECTED;
    linear_speed = 0.0;
    yaw_rate = 0.0;
    publish_stop_cmd = true;
    has_prev_error_ = false;
    filtered_deriv_ = 0.0;
  } else if (yellow_line_stop_active_ &&
             since_yellow_detection < yellow_stop_duration_)
  {
    current_state = TrackState::YELLOW_LINE_DETECTED;
    linear_speed = 0.0;
    yaw_rate = 0.0;
    publish_stop_cmd = true;
    has_prev_error_ = false;
    filtered_deriv_ = 0.0;
  } else {
    if (yellow_line_stop_active_ &&
        since_yellow_detection >= yellow_stop_duration_)
    {
      yellow_line_stop_active_ = false;
      RCLCPP_INFO(
        this->get_logger(),
        "Yellow line stop duration ended. Resuming line following.");
    }

    if (result_opt && result_opt->line_detected) {
      const auto & result = *result_opt;
      last_detection_time_ = now;
      current_state = TrackState::TRACKING;

      const double kd = std::max(0.0, this->get_parameter("kd").as_double());
      double deriv_term = 0.0;
      if (has_prev_error_) {
        const double dt = (now - last_error_time_).seconds();
        if (dt >= 0.005 && dt <= 0.25) {
          const double raw_deriv = (result.error - prev_error_) / dt;
          filtered_deriv_ = 0.55 * filtered_deriv_ + 0.45 * raw_deriv;
          deriv_term = kd * filtered_deriv_;
        } else {
          filtered_deriv_ = 0.0;
        }
      }
      prev_error_ = result.error;
      last_error_time_ = now;
      has_prev_error_ = true;

      const double yaw_cmd =
        clamp(-(kp_ * result.error + deriv_term), -max_yaw_rate_, max_yaw_rate_);

      const double composite_err = std::abs(result.error);
      const double speed_scale =
        1.0 - speed_gain_damping_ * std::min(1.0, composite_err);
      double speed_cmd =
        clamp(target_speed_ * speed_scale, min_speed_, max_speed_);

      if (this->get_parameter("enable_continuous_curve_assist").as_bool()) {
        const double reverse_curve_speed =
          std::max(0.0, this->get_parameter("reverse_curve_speed").as_double());
        const double sharp_curve_threshold =
          std::max(0.0, this->get_parameter("sharp_curve_threshold").as_double());
        const double sharp_curve_speed =
          std::max(0.0, this->get_parameter("sharp_curve_speed").as_double());

        if (g_curve_assist.reverse_curve_active) {
          speed_cmd = std::min(speed_cmd, reverse_curve_speed);
        } else if (g_curve_assist.curve_level >= sharp_curve_threshold) {
          speed_cmd = std::min(speed_cmd, sharp_curve_speed);
        }
      }

      linear_speed = speed_cmd;
      yaw_rate = yaw_cmd;
    } else {
      has_prev_error_ = false;
      filtered_deriv_ = 0.0;
      const double since_detection =
        (now - last_detection_time_).seconds();
      current_state =
        since_detection >= line_lost_stop_time_ ?
        TrackState::LOST : last_state_;

      linear_speed = line_lost_forward_speed;
      yaw_rate =
        g_motion.valid ?
        clamp(
          g_motion.last_yaw * line_lost_yaw_hold_factor,
          -max_yaw_rate_,
          max_yaw_rate_) :
        0.0;
      publish_stop_cmd = false;
    }
  }

  const bool yellow_stop_now =
    permanent_stop_ ||
    (yellow_line_stop_active_ && since_yellow_detection < yellow_stop_duration_);

  if (!yellow_stop_now && intersection_active_) {
    if (now <= intersection_end_time_) {
      linear_speed = intersection_turn_speed_;
      yaw_rate = clamp(intersection_turn_yaw_rate_, -max_yaw_rate_, max_yaw_rate_);
      current_state = TrackState::TRACKING;
    } else {
      intersection_active_ = false;
    }
  }

  // ============================================================
  // `┤` 丁字汇合口 DAG 顺行闭环并线状态机（与单行道 90° 弯 / S 弯完全正交）
  // ============================================================
  if (!yellow_stop_now && this->get_parameter("enable_tjunction_merge").as_bool()) {
    const bool require_gate =
      this->get_parameter("tjunction_require_branch_gate").as_bool();
    const bool gate_ok = !require_gate || tjunction_merge_armed_;
    const int confirm_frames = std::max(
      1, static_cast<int>(this->get_parameter("tjunction_confirm_frames").as_int()));
    const double merge_speed =
      std::max(0.05, this->get_parameter("tjunction_merge_speed").as_double());
    const double merge_yaw_mag =
      std::abs(this->get_parameter("tjunction_merge_yaw_rate").as_double());
    const int merge_dir =
      (this->get_parameter("tjunction_merge_direction").as_int() >= 0) ? 1 : -1;
    const double min_turn_dur =
      std::max(0.10, this->get_parameter("tjunction_min_turn_duration").as_double());
    const double max_turn_dur =
      std::max(min_turn_dur, this->get_parameter("tjunction_max_turn_duration").as_double());
    const double exit_upper_ratio = clamp(
      this->get_parameter("tjunction_exit_upper_road_ratio").as_double(), 0.02, 0.80);

    if (!tjunction_merging_) {
      if (gate_ok && result_opt && result_opt->tjunction_candidate) {
        tjunction_confirm_count_++;
        if (tjunction_confirm_count_ >= confirm_frames) {
          tjunction_merging_ = true;
          tjunction_merge_armed_ = false;
          tjunction_confirm_count_ = 0;
          tjunction_merge_start_time_ = now;
          RCLCPP_INFO(
            this->get_logger(),
            "[T-merge] Horizontal T-wall confirmed (full_rows=%d). Merging %s at v=%.2f, w=%+.2f.",
            result_opt->tjunction_full_rows,
            merge_dir < 0 ? "RIGHT (forward)" : "LEFT (forward)",
            merge_speed, merge_dir * merge_yaw_mag);
        }
      } else {
        tjunction_confirm_count_ = 0;
      }
    }

    if (tjunction_merging_) {
      const double elapsed = (now - tjunction_merge_start_time_).seconds();
      const bool road_open_ahead =
        result_opt &&
        elapsed >= min_turn_dur &&
        result_opt->upper_road_ratio >= exit_upper_ratio &&
        result_opt->tjunction_full_rows < 20;

      if (road_open_ahead || elapsed >= max_turn_dur) {
        tjunction_merging_ = false;
        has_prev_error_ = false;
        filtered_deriv_ = 0.0;
        RCLCPP_INFO(
          this->get_logger(),
          "[T-merge] Completed in %.2fs (upper_road=%.2f). Handing back to closed-loop PD.",
          elapsed, result_opt ? result_opt->upper_road_ratio : -1.0);
      } else {
        linear_speed = merge_speed;
        yaw_rate = clamp(merge_dir * merge_yaw_mag, -max_yaw_rate_, max_yaw_rate_);
        current_state = TrackState::TRACKING;
      }
    }
  }

  // ============================================================
  // 动态红锥 / 红色禁行区软避障融合
  // 若正处于 90° 丁字并线转向中，仅当极近贴脸障碍（very_close）时才打断并线角速度，
  // 防止汇合口下游远端禁行区在完成 90° 右转前反向推车；并线完成后立即恢复全距离避障。
  // ============================================================
  bool publish_avoid_cmd = false;

  if (!yellow_stop_now && soft_red_enabled) {
    const double bottom_ratio = clamp(g_red_avoid.bottom_ratio, 0.0, 1.0);
    const bool allow_during_merge =
      !tjunction_merging_ || (bottom_ratio >= red_very_close_ratio);

    if (soft_red_active && allow_during_merge) {
      const double risk_denominator =
        std::max(1e-3, red_very_close_ratio - active_red_near_ratio);
      const double risk =
        clamp((bottom_ratio - active_red_near_ratio) / risk_denominator, 0.0, 1.0);
      g_red_avoid.risk = risk;

      const double repulsion_gain =
        std::max(0.0, this->get_parameter("red_repulsion_gain").as_double());
      const double lane_keep_min_factor =
        clamp(this->get_parameter("red_lane_keep_min_factor").as_double(), 0.0, 1.0);

      const bool red_on_left = soft_red_yaw_bias < 0.0;
      double repulsion_strength =
        std::abs(soft_red_yaw_bias) * (1.0 + repulsion_gain * risk);

      if (g_curve_assist.reverse_curve_active && !g_red_avoid.is_red_line) {
        const double curve_bias_limit =
          clamp(this->get_parameter("red_soft_curve_yaw_limit").as_double(), 0.0, max_yaw_rate_);
        const double dynamic_limit =
          curve_bias_limit + risk * (max_yaw_rate_ - curve_bias_limit);
        repulsion_strength = std::min(repulsion_strength, dynamic_limit);
      }

      const double lane_keep_factor =
        1.0 - (1.0 - lane_keep_min_factor) * risk;

      if (red_on_left) {
        if (yaw_rate > 0.0) {
          yaw_rate *= lane_keep_factor;
        }
        yaw_rate -= repulsion_strength;
      } else {
        if (yaw_rate < 0.0) {
          yaw_rate *= lane_keep_factor;
        }
        yaw_rate += repulsion_strength;
      }

      yaw_rate = clamp(yaw_rate, -max_yaw_rate_, max_yaw_rate_);

      const double red_base_speed =
        g_red_avoid.is_red_line ?
        std::max(0.0, this->get_parameter("red_line_speed").as_double()) :
        red_soft_speed;
      const double safe_speed =
        std::max(red_min_forward_speed, red_base_speed * (1.0 - 0.60 * risk));

      linear_speed =
        std::max(red_min_forward_speed, std::min(linear_speed, safe_speed));

      g_red_avoid.very_close = bottom_ratio >= red_very_close_ratio;
      current_state = TrackState::TRACKING;

      // 当近距风险达到抢占阈值时，同步输出到 /cmd_vel_avoid，抢占数字开环转向拖尾防撞锥
      if (risk >= red_mux_preempt_risk) {
        publish_avoid_cmd = true;
      }
    }
  } else if (!yellow_stop_now && red_turning_ && now < red_turn_end_time_) {
    linear_speed = std::max(red_min_forward_speed, red_turn_linear_);
    yaw_rate = red_turn_yaw_cmd_;
    current_state = TrackState::TRACKING;
  } else if (red_turning_ && now >= red_turn_end_time_) {
    red_turning_ = false;
  }

  // ============================================================
  // 最终安全锁：黄色停车拥有最高优先级
  // ============================================================
  if (permanent_stop_ ||
      (yellow_line_stop_active_ && since_yellow_detection < yellow_stop_duration_))
  {
    current_state = TrackState::YELLOW_LINE_DETECTED;
    linear_speed = 0.0;
    yaw_rate = 0.0;
    publish_stop_cmd = true;
    publish_avoid_cmd = false;
  }

  if (!yellow_stop_now && linear_speed > 0.0) {
    g_motion.valid = true;
    g_motion.last_linear = linear_speed;
    g_motion.last_yaw = yaw_rate;
  }

  if (result_opt && debug_enabled_) {
    publish_debug_image(msg, frame_bgr, *result_opt);
  }

  publish_cmd_vel(linear_speed, yaw_rate, publish_stop_cmd, publish_avoid_cmd);

  if (current_state != last_state_ ||
      current_state == TrackState::LOST ||
      current_state == TrackState::YELLOW_LINE_DETECTED)
  {
    publish_state(current_state);
  }

  last_state_ = current_state;
}

std::optional<ProcessingResult> LineFollowerNode::process_image(
  const cv::Mat & frame_bgr) const
{
  if (frame_bgr.empty()) {
    return std::nullopt;
  }

  const int height = frame_bgr.rows;
  const int width = frame_bgr.cols;

  const int roi_top = std::clamp(
    static_cast<int>(height * (1.0 - roi_top_ratio_)),
    0,
    height - 1);

  const int roi_bottom = std::clamp(
    height - static_cast<int>(height * roi_bottom_ratio_),
    roi_top + 1,
    height);

  const int roi_height = roi_bottom - roi_top;
  const cv::Rect roi_rect(0, roi_top, width, roi_height);
  const cv::Mat roi = frame_bgr(roi_rect);

  cv::Mat blurred;
  cv::GaussianBlur(
    roi,
    blurred,
    cv::Size(gaussian_kernel_, gaussian_kernel_),
    0);

  cv::Mat hsv;
  cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

  bool yellow_line_detected = false;
  cv::Mat yellow_mask;

  if (use_yellow_filter_) {
    const cv::Scalar lower_yellow(
      static_cast<double>(yellow_h_low_),
      static_cast<double>(yellow_s_min_),
      static_cast<double>(yellow_v_min_));
    const cv::Scalar upper_yellow(
      static_cast<double>(yellow_h_high_),
      255.0,
      255.0);

    cv::inRange(hsv, lower_yellow, upper_yellow, yellow_mask);

    const int yellow_pixels = cv::countNonZero(yellow_mask);
    const double yellow_area_ratio =
      yellow_mask.total() > 0 ?
      static_cast<double>(yellow_pixels) / static_cast<double>(yellow_mask.total()) :
      0.0;

    yellow_line_detected =
      yellow_area_ratio >= yellow_area_ratio_threshold_;
  }

  cv::Mat binary;
  if (use_white_filter_) {
    const cv::Scalar lower_white(
      static_cast<double>(white_h_low_),
      0.0,
      static_cast<double>(white_v_min_));
    const cv::Scalar upper_white(
      static_cast<double>(white_h_high_),
      static_cast<double>(white_s_max_),
      255.0);
    cv::inRange(hsv, lower_white, upper_white, binary);
  } else {
    cv::Mat gray;
    cv::cvtColor(blurred, gray, cv::COLOR_BGR2GRAY);
    const int threshold_type =
      invert_binary_ ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY;
    cv::threshold(gray, binary, binary_threshold_, 255, threshold_type);
  }

  cv::morphologyEx(
    binary, binary, cv::MORPH_OPEN, morph_element_, cv::Point(-1, -1), 1);
  cv::morphologyEx(
    binary, binary, cv::MORPH_CLOSE, morph_element_, cv::Point(-1, -1), morph_iterations_);

  ProcessingResult result{};
  result.line_detected = false;
  result.yellow_line_detected = yellow_line_detected;
  result.roi_bgr = roi.clone();
  result.binary_mask = binary.clone();
  result.yellow_mask = yellow_mask;
  result.roi_top = roi_rect.y;
  result.roi_bottom = roi_rect.y + roi_rect.height;

  // ============================================================
  // 若启用丁字并线检测，提取车底种子 8 连通道路掩膜并计算水平直墙与全宽横贯行指标
  // ============================================================
  if (this->get_parameter("enable_tjunction_merge").as_bool() &&
      roi_height >= 40 && width >= 100)
  {
    cv::Mat bridged;
    if (!yellow_mask.empty() && yellow_mask.size() == binary.size()) {
      cv::bitwise_or(binary, yellow_mask, bridged);
    } else {
      bridged = binary;
    }

    cv::Mat labels;
    const int num_labels = cv::connectedComponents(bridged, labels, 8, CV_32S);
    std::vector<unsigned char> keep_label(static_cast<size_t>(std::max(1, num_labels)), 0);

    const int seed_y0 = std::clamp(static_cast<int>(roi_height * 0.78), 0, roi_height - 1);
    const int seed_x0 = std::clamp(static_cast<int>(width * 0.15), 0, width - 1);
    const int seed_x1 = std::clamp(static_cast<int>(width * 0.85), seed_x0 + 1, width);

    bool has_seed = false;
    for (int y = seed_y0; y < roi_height; ++y) {
      const int * row_ptr = labels.ptr<int>(y);
      for (int x = seed_x0; x < seed_x1; ++x) {
        const int lb = row_ptr[x];
        if (lb > 0) {
          keep_label[static_cast<size_t>(lb)] = 1;
          has_seed = true;
        }
      }
    }

    cv::Mat connected_road = cv::Mat::zeros(binary.size(), CV_8UC1);
    if (has_seed) {
      for (int y = 0; y < roi_height; ++y) {
        const int * lb_ptr = labels.ptr<int>(y);
        const unsigned char * bin_ptr = binary.ptr<unsigned char>(y);
        unsigned char * out_ptr = connected_road.ptr<unsigned char>(y);
        for (int x = 0; x < width; ++x) {
          if (bin_ptr[x] > 0 && keep_label[static_cast<size_t>(lb_ptr[x])] != 0) {
            out_ptr[x] = 255;
          }
        }
      }
    }

    // 顶部前视区白路占比（用于闭环退出判定）
    const int up_h = std::max(1, static_cast<int>(roi_height * 0.35));
    const cv::Mat upper_patch = connected_road(cv::Rect(seed_x0, 0, seed_x1 - seed_x0, up_h));
    result.upper_road_ratio =
      upper_patch.total() > 0 ?
      static_cast<double>(cv::countNonZero(upper_patch)) /
        static_cast<double>(upper_patch.total()) :
      0.0;

    if (!yellow_line_detected && has_seed) {
      // 5 个横向切片检测道路上沿 y_top（按 640 基准宽度等比缩放）
      const std::array<std::pair<double, double>, 5> kSlices{{
        {20.0 / 640.0, 120.0 / 640.0},
        {140.0 / 640.0, 240.0 / 640.0},
        {270.0 / 640.0, 370.0 / 640.0},
        {400.0 / 640.0, 500.0 / 640.0},
        {520.0 / 640.0, 620.0 / 640.0},
      }};

      bool all_slices_valid = true;
      int min_yt = roi_height;
      int max_yt = 0;

      for (const auto & sl : kSlices) {
        const int x0 = std::clamp(static_cast<int>(std::round(sl.first * width)), 0, width - 2);
        const int x1 = std::clamp(static_cast<int>(std::round(sl.second * width)), x0 + 1, width);
        const int min_slice_white = std::max(5, static_cast<int>((x1 - x0) * 0.50));
        int found_y = -1;
        for (int y = 0; y < roi_height; ++y) {
          const unsigned char * rptr = connected_road.ptr<unsigned char>(y);
          int cnt = 0;
          for (int x = x0; x < x1; ++x) {
            cnt += (rptr[x] > 0) ? 1 : 0;
          }
          if (cnt >= min_slice_white) {
            found_y = y;
            break;
          }
        }
        if (found_y < 0) {
          all_slices_valid = false;
          break;
        }
        min_yt = std::min(min_yt, found_y);
        max_yt = std::max(max_yt, found_y);
      }

      if (all_slices_valid) {
        const int spread = max_yt - min_yt;
        const int y_start = std::min(roi_height - 1, max_yt + 4);
        const int y_end = std::clamp(static_cast<int>(roi_height * 0.94), y_start + 1, roi_height);
        const int edge_w = std::max(4, static_cast<int>(std::round(24.0 * width / 640.0)));
        const int edge_min = std::max(3, static_cast<int>(std::round(18.0 * width / 640.0)));
        const int full_row_min = static_cast<int>(std::round(0.94 * width));

        int full_rows = 0;
        for (int y = y_start; y < y_end; ++y) {
          const unsigned char * rptr = connected_road.ptr<unsigned char>(y);
          int total_w = 0;
          int left_w = 0;
          int right_w = 0;
          for (int x = 0; x < width; ++x) {
            if (rptr[x] > 0) {
              total_w++;
              if (x < edge_w) {
                left_w++;
              }
              if (x >= width - edge_w) {
                right_w++;
              }
            }
          }
          if (total_w >= full_row_min && left_w >= edge_min && right_w >= edge_min) {
            full_rows++;
          }
        }
        result.tjunction_full_rows = full_rows;

        const double wall_min_y =
          this->get_parameter("tjunction_wall_min_y_ratio").as_double() * roi_height;
        const double wall_max_y =
          this->get_parameter("tjunction_wall_max_y_ratio").as_double() * roi_height;
        const double wall_max_spread =
          this->get_parameter("tjunction_wall_max_spread_ratio").as_double() * roi_height;
        const int req_full_rows =
          static_cast<int>(this->get_parameter("tjunction_min_full_rows").as_int());

        if (min_yt >= static_cast<int>(wall_min_y) &&
            max_yt <= static_cast<int>(wall_max_y) &&
            spread <= static_cast<int>(wall_max_spread) &&
            (y_end - y_start) >= 45 &&
            full_rows >= req_full_rows)
        {
          result.tjunction_candidate = true;
        }
      }
    }
  }

  // ============================================================
  // 最大轮廓只做存在性判断 / fallback / debug
  // ============================================================
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  bool contour_valid = false;
  cv::Point2f contour_centroid(
    static_cast<float>(width) * 0.5F,
    static_cast<float>(roi_height) * 0.70F);

  if (!contours.empty()) {
    double max_area = 0.0;
    size_t best_index = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
      const double area = cv::contourArea(contours[i]);
      if (area > max_area) {
        max_area = area;
        best_index = i;
      }
    }

    if (max_area >= min_contour_area_) {
      contour_valid = true;
      result.bounding_box = cv::boundingRect(contours[best_index]);
      const auto moments = cv::moments(contours[best_index]);
      if (moments.m00 != 0.0) {
        contour_centroid.x = static_cast<float>(moments.m10 / moments.m00);
        contour_centroid.y = static_cast<float>(moments.m01 / moments.m00);
      } else {
        contour_centroid.x =
          static_cast<float>(result.bounding_box.x + result.bounding_box.width / 2.0);
        contour_centroid.y =
          static_cast<float>(result.bounding_box.y + result.bounding_box.height / 2.0);
      }
    }
  }

  // ============================================================
  // 局部前视白色通道中心（返回通道中心与宽度）
  // ============================================================
  auto find_band_run =
    [&](int y0, int y1, double reference_x) -> std::optional<BandRun>
    {
      y0 = std::clamp(y0, 0, binary.rows - 1);
      y1 = std::clamp(y1, y0 + 1, binary.rows);
      const cv::Mat band = binary(cv::Rect(0, y0, binary.cols, y1 - y0));
      const int band_height = band.rows;
      if (band_height <= 0) {
        return std::nullopt;
      }

      constexpr double kColumnWhiteRatio = 0.30;
      const int min_white_pixels = std::max(
        1, static_cast<int>(static_cast<double>(band_height) * kColumnWhiteRatio));

      std::vector<unsigned char> active(binary.cols, 0);
      for (int x = 0; x < binary.cols; ++x) {
        if (cv::countNonZero(band.col(x)) >= min_white_pixels) {
          active[x] = 1;
        }
      }

      constexpr int kMinRunWidth = 20;
      bool found = false;
      double best_center = reference_x;
      double best_distance = 1e9;
      int best_width = 0;
      int run_start = -1;

      for (int x = 0; x <= binary.cols; ++x) {
        const bool is_active = (x < binary.cols) ? (active[x] != 0) : false;
        if (is_active && run_start < 0) {
          run_start = x;
        }
        if ((!is_active || x == binary.cols) && run_start >= 0) {
          const int run_end = x - 1;
          const int run_width = run_end - run_start + 1;
          if (run_width >= kMinRunWidth) {
            const double center = 0.5 * static_cast<double>(run_start + run_end);
            double distance =
              (reference_x >= static_cast<double>(run_start) &&
               reference_x <= static_cast<double>(run_end)) ?
              0.0 :
              std::abs(center - reference_x);

            if (!found ||
                distance < best_distance ||
                (std::abs(distance - best_distance) < 1e-6 && run_width > best_width))
            {
              found = true;
              best_center = center;
              best_distance = distance;
              best_width = run_width;
            }
          }
          run_start = -1;
        }
      }

      if (!found) {
        return std::nullopt;
      }
      return BandRun{best_center, best_width};
    };

  const int far_y0 = static_cast<int>(static_cast<double>(roi_height) * 0.34);
  const int far_y1 = static_cast<int>(static_cast<double>(roi_height) * 0.56);
  const int near_y0 = static_cast<int>(static_cast<double>(roi_height) * 0.62);
  const int near_y1 = static_cast<int>(static_cast<double>(roi_height) * 0.88);
  const double image_center_x = static_cast<double>(width) * 0.5;

  const auto near_run = find_band_run(near_y0, near_y1, image_center_x);
  const double far_reference = near_run.has_value() ? near_run->center : image_center_x;
  const auto far_run = find_band_run(far_y0, far_y1, far_reference);

  const bool curve_assist_enabled =
    this->get_parameter("enable_continuous_curve_assist").as_bool();
  const bool preview_enabled =
    this->get_parameter("enable_curve_preview_steering").as_bool();
  const double preview_max_far_weight = clamp(
    this->get_parameter("curve_preview_max_far_weight").as_double(), 0.30, 0.65);
  const double reverse_curve_min_error =
    std::max(0.0, this->get_parameter("reverse_curve_min_error").as_double());
  const double reverse_curve_threshold =
    std::max(0.0, this->get_parameter("reverse_curve_threshold").as_double());
  const int reverse_curve_confirm_frames = std::max(
    1, static_cast<int>(this->get_parameter("reverse_curve_confirm_frames").as_int()));
  const int reverse_curve_clear_frames = std::max(
    1, static_cast<int>(this->get_parameter("reverse_curve_clear_frames").as_int()));

  const bool near_valid = near_run.has_value();
  const bool far_valid = far_run.has_value();

  double near_error = 0.0;
  double far_error = 0.0;
  if (near_valid) {
    near_error = clamp((near_run->center - image_center_x) / image_center_x, -1.0, 1.0);
  }
  if (far_valid) {
    far_error = clamp((far_run->center - image_center_x) / image_center_x, -1.0, 1.0);
  }

  g_curve_assist.near_valid = near_valid;
  g_curve_assist.far_valid = far_valid;
  g_curve_assist.near_error = near_error;
  g_curve_assist.far_error = far_error;
  g_curve_assist.curve_level = 0.0;
  g_curve_assist.far_weight = 0.30;

  if (near_valid) {
    g_curve_assist.curve_level = std::max(g_curve_assist.curve_level, std::abs(near_error));
  }
  if (far_valid) {
    g_curve_assist.curve_level = std::max(g_curve_assist.curve_level, std::abs(far_error));
  }

  bool reverse_curve_candidate = false;
  if (curve_assist_enabled && near_valid && far_valid) {
    const bool opposite_direction = near_error * far_error < 0.0;
    const bool both_significant =
      std::abs(near_error) >= reverse_curve_min_error &&
      std::abs(far_error) >= reverse_curve_min_error;
    const bool large_difference =
      std::abs(far_error - near_error) >= reverse_curve_threshold;
    reverse_curve_candidate =
      opposite_direction && both_significant && large_difference;
  }

  if (!curve_assist_enabled) {
    g_curve_assist.reverse_curve_active = false;
    g_curve_assist.reverse_confirm_count = 0;
    g_curve_assist.reverse_clear_count = 0;
  } else if (reverse_curve_candidate) {
    g_curve_assist.reverse_clear_count = 0;
    g_curve_assist.reverse_confirm_count++;
    if (!g_curve_assist.reverse_curve_active &&
        g_curve_assist.reverse_confirm_count >= reverse_curve_confirm_frames)
    {
      g_curve_assist.reverse_curve_active = true;
      g_curve_assist.reverse_confirm_count = reverse_curve_confirm_frames;
    }
  } else {
    g_curve_assist.reverse_confirm_count = 0;
    if (g_curve_assist.reverse_curve_active) {
      g_curve_assist.reverse_clear_count++;
      if (g_curve_assist.reverse_clear_count >= reverse_curve_clear_frames) {
        g_curve_assist.reverse_curve_active = false;
        g_curve_assist.reverse_clear_count = 0;
      }
    } else {
      g_curve_assist.reverse_clear_count = 0;
    }
  }

  // ============================================================
  // 生成最终目标点（兼容原 70/30，支持急弯动态远端预瞄）
  // ============================================================
  double target_x = static_cast<double>(contour_centroid.x);
  double target_y = static_cast<double>(contour_centroid.y);
  bool local_target_valid = false;

  if (near_valid && far_valid) {
    const double near_y = 0.5 * static_cast<double>(near_y0 + near_y1);
    const double far_y = 0.5 * static_cast<double>(far_y0 + far_y1);

    if (curve_assist_enabled && g_curve_assist.reverse_curve_active &&
        (!preview_enabled || near_error * far_error < 0.0))
    {
      target_x = near_run->center;
      target_y = near_y;
      g_curve_assist.far_weight = 0.0;
    } else {
      double far_weight = 0.30;
      if (preview_enabled) {
        if (near_error * far_error < 0.0) {
          far_weight =
            preview_max_far_weight * (1.0 - clamp(std::abs(near_error) / 0.25, 0.0, 1.0));
          if (reverse_curve_candidate) {
            far_weight *= 1.0 - clamp(
              static_cast<double>(g_curve_assist.reverse_confirm_count) /
                static_cast<double>(reverse_curve_confirm_frames),
              0.0, 1.0);
          }
        } else {
          far_weight +=
            (preview_max_far_weight - 0.30) *
            clamp(std::max(0.0, std::abs(far_error) - std::abs(near_error)) / 0.20, 0.0, 1.0);
          // 当近端处于宽阔路口咽喉(>=90%全宽)而远端已收敛进单侧弯道(<80%全宽)时，避免近端稀释远端转向
          if (near_run->width >= static_cast<int>(0.90 * width) &&
              far_run->width < static_cast<int>(0.80 * width))
          {
            far_weight = std::max(far_weight, preview_max_far_weight);
          }
        }
      }
      g_curve_assist.far_weight = far_weight;
      target_x = (1.0 - far_weight) * near_run->center + far_weight * far_run->center;
      // 保持纵向参考点稳定为 0.70*near_y + 0.30*far_y，防止弯道预瞄改变 centroid_y 影响数字对齐时机
      target_y = 0.70 * near_y + 0.30 * far_y;
    }
    local_target_valid = true;
  } else if (near_valid) {
    target_x = near_run->center;
    target_y = 0.5 * static_cast<double>(near_y0 + near_y1);
    g_curve_assist.far_weight = 0.0;
    local_target_valid = true;
  } else if (far_valid) {
    target_x = far_run->center;
    target_y = 0.5 * static_cast<double>(far_y0 + far_y1);
    g_curve_assist.far_weight = 1.0;
    local_target_valid = true;
  }

  if (!local_target_valid && !contour_valid) {
    return result;
  }

  result.line_detected = true;
  result.centroid = cv::Point2f(static_cast<float>(target_x), static_cast<float>(target_y));
  result.error = clamp((target_x - image_center_x) / image_center_x, -1.0, 1.0);
  result.state = TrackState::TRACKING;

  return result;
}

bool LineFollowerNode::detect_red(
  const cv::Mat & frame_bgr, int roi_top, int roi_bottom,
  const cv::Mat & road_mask, const cv::Point2d & line_centroid_roi, double & side_sign,
  cv::Point2d & red_centroid_img, cv::Point2d & div_base_img,
  cv::Point2d & div_target_img) const
{
  side_sign = 0.0;
  g_red_avoid = RedAvoidRuntime{};

  if (!enable_red_filter_ || frame_bgr.empty()) {
    return false;
  }

  roi_top = std::clamp(roi_top, 0, frame_bgr.rows - 1);
  roi_bottom = std::clamp(roi_bottom, roi_top + 1, frame_bgr.rows);
  const cv::Rect roi_rect(0, roi_top, frame_bgr.cols, roi_bottom - roi_top);
  const cv::Mat roi = frame_bgr(roi_rect);

  cv::Mat hsv;
  cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

  // ============================================================
  // 分支 1：保留原版 3x3 开闭运算实心红锥与实心横线检测
  // ============================================================
  cv::Mat mask1, mask2, mask;
  cv::inRange(
    hsv,
    cv::Scalar(static_cast<double>(red_h_low_1_), static_cast<double>(red_s_min_), static_cast<double>(red_v_min_)),
    cv::Scalar(static_cast<double>(red_h_high_1_), 255.0, 255.0),
    mask1);
  cv::inRange(
    hsv,
    cv::Scalar(static_cast<double>(red_h_low_2_), static_cast<double>(red_s_min_), static_cast<double>(red_v_min_)),
    cv::Scalar(static_cast<double>(red_h_high_2_), 255.0, 255.0),
    mask2);
  cv::bitwise_or(mask1, mask2, mask);

  const cv::Mat red_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, red_kernel, cv::Point(-1, -1), 1);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, red_kernel, cv::Point(-1, -1), 1);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  const double line_aspect_min =
    std::max(1.0, this->get_parameter("red_line_aspect_ratio_min").as_double());
  const double line_width_ratio_min =
    clamp(this->get_parameter("red_line_min_width_ratio").as_double(), 0.0, 1.0);
  const double cone_area_ratio_min =
    std::max(0.0, this->get_parameter("red_cone_area_ratio_threshold").as_double());
  const int cone_min_bbox_width =
    std::max(1, static_cast<int>(this->get_parameter("red_cone_min_bbox_width").as_int()));
  const int cone_min_bbox_height =
    std::max(1, static_cast<int>(this->get_parameter("red_cone_min_bbox_height").as_int()));

  const double roi_area = std::max(1.0, static_cast<double>(mask.total()));

  bool has_cone = false;
  double best_cone_area = 0.0;
  cv::Rect best_cone_box{};

  bool has_line = false;
  double best_line_area = 0.0;
  cv::Rect best_line_box{};

  for (size_t i = 0; i < contours.size(); ++i) {
    const double area = cv::contourArea(contours[i]);
    if (area <= 0.0) {
      continue;
    }
    const cv::Rect box = cv::boundingRect(contours[i]);
    if (box.width <= 0 || box.height <= 0) {
      continue;
    }

    const double aspect_ratio =
      static_cast<double>(box.width) / static_cast<double>(box.height);
    const bool looks_like_red_line =
      aspect_ratio >= line_aspect_min &&
      static_cast<double>(box.width) >= static_cast<double>(frame_bgr.cols) * line_width_ratio_min;

    if (looks_like_red_line) {
      if (!has_line || area > best_line_area) {
        has_line = true;
        best_line_area = area;
        best_line_box = box;
      }
      continue;
    }

    if (box.width < cone_min_bbox_width || box.height < cone_min_bbox_height) {
      continue;
    }
    if ((area / roi_area) < cone_area_ratio_min) {
      continue;
    }
    if (!has_cone || area > best_cone_area) {
      has_cone = true;
      best_cone_area = area;
      best_cone_box = box;
    }
  }

  // ============================================================
  // 分支 2：镂空红色禁行区（1cm 细框 + 多竖纹）闭运算聚类检测
  // 校验条件：地面扁平框(aspect>=1.30)、低填充率(0.025~max_fill)、至少 2 条独立红条纹、
  //          且处于白色赛道面上(white_ratio>=0.50)，彻底滤除红锥与场外红色杂物。
  // ============================================================
  if (this->get_parameter("enable_red_zone_cluster").as_bool() && !road_mask.empty()) {
    const int zone_s_min = std::min(70, red_s_min_);
    const int zone_v_min = std::min(60, red_v_min_);
    cv::Mat zm1, zm2, raw_zone;
    cv::inRange(
      hsv,
      cv::Scalar(0.0, static_cast<double>(zone_s_min), static_cast<double>(zone_v_min)),
      cv::Scalar(12.0, 255.0, 255.0),
      zm1);
    cv::inRange(
      hsv,
      cv::Scalar(158.0, static_cast<double>(zone_s_min), static_cast<double>(zone_v_min)),
      cv::Scalar(180.0, 255.0, 255.0),
      zm2);
    cv::bitwise_or(zm1, zm2, raw_zone);

    cv::Mat labels, stats, centroids;
    const int num_cc =
      cv::connectedComponentsWithStats(raw_zone, labels, stats, centroids, 8, CV_32S);
    cv::Mat clean_raw = cv::Mat::zeros(raw_zone.size(), CV_8UC1);
    std::vector<cv::Rect> stripe_boxes;
    for (int idx = 1; idx < num_cc; ++idx) {
      const int cc_area = stats.at<int>(idx, cv::CC_STAT_AREA);
      if (cc_area >= 6) {
        stripe_boxes.emplace_back(
          stats.at<int>(idx, cv::CC_STAT_LEFT),
          stats.at<int>(idx, cv::CC_STAT_TOP),
          stats.at<int>(idx, cv::CC_STAT_WIDTH),
          stats.at<int>(idx, cv::CC_STAT_HEIGHT));
      }
    }

    if (stripe_boxes.size() >= 2) {
      for (int y = 0; y < raw_zone.rows; ++y) {
        const int * lb_ptr = labels.ptr<int>(y);
        unsigned char * out_ptr = clean_raw.ptr<unsigned char>(y);
        for (int x = 0; x < raw_zone.cols; ++x) {
          const int lb = lb_ptr[x];
          if (lb > 0 && stats.at<int>(lb, cv::CC_STAT_AREA) >= 6) {
            out_ptr[x] = 255;
          }
        }
      }

      const int kw = std::max(
        5, static_cast<int>(this->get_parameter("red_zone_close_kernel_w").as_int()));
      const int kh = std::max(
        3, static_cast<int>(this->get_parameter("red_zone_close_kernel_h").as_int()));
      const cv::Mat zone_kernel =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kw, kh));
      cv::Mat clustered;
      cv::morphologyEx(clean_raw, clustered, cv::MORPH_CLOSE, zone_kernel);

      const double min_w_ratio =
        clamp(this->get_parameter("red_zone_min_width_ratio").as_double(), 0.05, 0.90);
      const double min_area_ratio =
        std::max(0.0005, this->get_parameter("red_zone_min_area_ratio").as_double());
      const double max_fill_ratio =
        clamp(this->get_parameter("red_zone_max_fill_ratio").as_double(), 0.08, 0.60);

      std::vector<std::vector<cv::Point>> zone_contours;
      cv::findContours(clustered, zone_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

      for (const auto & zc : zone_contours) {
        const cv::Rect zb = cv::boundingRect(zc);
        if (zb.width < static_cast<int>(roi.cols * min_w_ratio) || zb.height < 18) {
          continue;
        }
        if (static_cast<double>(zb.width) / static_cast<double>(zb.height) < 1.30) {
          continue;
        }
        const double box_area = static_cast<double>(zb.width * zb.height);
        if ((box_area / roi_area) < min_area_ratio) {
          continue;
        }
        const double raw_nz = static_cast<double>(cv::countNonZero(clean_raw(zb)));
        const double raw_fill = raw_nz / std::max(1.0, box_area);
        if (raw_fill < 0.025 || raw_fill > max_fill_ratio) {
          continue;
        }

        int stripes_inside = 0;
        for (const auto & sb : stripe_boxes) {
          if (sb.x >= zb.x - 2 && sb.y >= zb.y - 2 &&
              sb.x + sb.width <= zb.x + zb.width + 2 &&
              sb.y + sb.height <= zb.y + zb.height + 2)
          {
            stripes_inside++;
          }
        }
        if (stripes_inside < 2) {
          continue;
        }

        const int px0 = std::max(0, zb.x - 12);
        const int py0 = std::max(0, zb.y - 8);
        const int px1 = std::min(road_mask.cols, zb.x + zb.width + 12);
        const int py1 = std::min(road_mask.rows, zb.y + zb.height + 8);
        if (px1 > px0 && py1 > py0) {
          const cv::Mat road_patch = road_mask(cv::Rect(px0, py0, px1 - px0, py1 - py0));
          const double white_ratio =
            static_cast<double>(cv::countNonZero(road_patch)) /
            static_cast<double>(road_patch.total());
          if (white_ratio < 0.50) {
            continue;
          }
        }

        if (!has_line || box_area > best_line_area) {
          has_line = true;
          best_line_area = box_area;
          best_line_box = zb;
        }
      }

      // 若原分支检出的所谓“小锥”只是禁行区内部的一根细竖纹碎片，提升为整块禁行区处理
      if (has_line && has_cone) {
        if (best_cone_box.x >= best_line_box.x - 6 &&
            best_cone_box.y >= best_line_box.y - 6 &&
            best_cone_box.x + best_cone_box.width <= best_line_box.x + best_line_box.width + 6 &&
            best_cone_box.y + best_cone_box.height <= best_line_box.y + best_line_box.height + 6)
        {
          has_cone = false;
        }
      }
    }
  }

  if (has_line) {
    g_red_avoid.red_line_detected = true;
    g_red_avoid.red_line_bbox_img = cv::Rect(
      best_line_box.x,
      best_line_box.y + roi_top,
      best_line_box.width,
      best_line_box.height);
  }

  // 当禁行区与远端红锥同框时，按距离车头更近者优先（避免远处无害红锥遮蔽脚下禁行区）
  const int cone_bottom = best_cone_box.y + best_cone_box.height;
  const int line_bottom = best_line_box.y + best_line_box.height;
  const bool use_red_line =
    has_line && (!has_cone || line_bottom >= cone_bottom - 15);
  if (!has_cone && !has_line) {
    return false;
  }

  const cv::Rect box = use_red_line ? best_line_box : best_cone_box;
  const double selected_area = use_red_line ? best_line_area : best_cone_area;
  const double aspect_ratio =
    static_cast<double>(box.width) / static_cast<double>(std::max(1, box.height));
  const double contour_area_ratio = selected_area / roi_area;

  const cv::Point2d red_centroid_roi(
    static_cast<double>(box.x) + static_cast<double>(box.width) * 0.5,
    static_cast<double>(box.y) + static_cast<double>(box.height) * 0.5);

  cv::Point2d red_bottom_roi(
    static_cast<double>(box.x) + static_cast<double>(box.width) * 0.5,
    static_cast<double>(box.y + box.height));

  cv::Point2d base(
    static_cast<double>(mask.cols) * 0.5,
    static_cast<double>(mask.rows));

  cv::Point2d v_line = line_centroid_roi - base;
  if (red_divider_angle_rad_ != 0.0) {
    const double angle =
      v_line.x >= 0.0 ? red_divider_angle_rad_ : -red_divider_angle_rad_;
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    v_line = cv::Point2d(
      v_line.x * c - v_line.y * s,
      v_line.x * s + v_line.y * c);
  }

  if (use_red_line) {
    // 基于禁行区左右两侧“真实白色可通行路面像素”选择绕行侧，绝不把蓝垫误判为空余空间
    double left_road_space = static_cast<double>(box.x);
    double right_road_space = static_cast<double>(mask.cols - (box.x + box.width));

    if (!road_mask.empty() && road_mask.size() == mask.size()) {
      const int by0 = std::clamp(box.y - 10, 0, road_mask.rows - 1);
      const int by1 = std::clamp(box.y + box.height + 10, by0 + 1, road_mask.rows);
      const int band_h = std::max(1, by1 - by0);
      left_road_space =
        box.x > 0 ?
        static_cast<double>(cv::countNonZero(road_mask(cv::Rect(0, by0, box.x, band_h)))) /
          static_cast<double>(band_h) :
        0.0;
      const int rx0 = std::clamp(box.x + box.width, 0, road_mask.cols);
      right_road_space =
        rx0 < road_mask.cols ?
        static_cast<double>(
          cv::countNonZero(road_mask(cv::Rect(rx0, by0, road_mask.cols - rx0, band_h)))) /
          static_cast<double>(band_h) :
        0.0;
    }

    if (right_road_space > left_road_space + 12.0) {
      side_sign = -1.0;  // 右侧白路更宽 -> 向右绕（等价于障碍在左）
    } else if (left_road_space > right_road_space + 12.0) {
      side_sign = 1.0;   // 左侧白路更宽 -> 向左绕
    } else {
      const int fallback_direction =
        static_cast<int>(this->get_parameter("red_line_fallback_direction").as_int());
      side_sign = fallback_direction >= 0 ? 1.0 : -1.0;
    }

    // 用禁行区伸入可通行车道一侧的内侧边角计算走廊距离，防止中心偏在边上漏避让
    red_bottom_roi.x =
      (side_sign < 0.0) ?
      static_cast<double>(std::min(mask.cols - 1, box.x + box.width)) :
      static_cast<double>(std::max(0, box.x));
  } else {
    const cv::Point2d v_red = red_centroid_roi - base;
    const double cross = v_line.x * v_red.y - v_line.y * v_red.x;
    side_sign = cross >= 0.0 ? 1.0 : -1.0;
  }

  red_centroid_img = cv::Point2d(
    red_centroid_roi.x,
    red_centroid_roi.y + static_cast<double>(roi_top));

  const cv::Point2d red_bottom_img(
    red_bottom_roi.x,
    red_bottom_roi.y + static_cast<double>(roi_top));

  div_base_img = cv::Point2d(
    base.x,
    base.y + static_cast<double>(roi_top));

  const cv::Point2d rotated_target = base + v_line;
  div_target_img = cv::Point2d(
    rotated_target.x,
    rotated_target.y + static_cast<double>(roi_top));

  g_red_avoid.valid = true;
  g_red_avoid.is_red_line = use_red_line;
  g_red_avoid.bbox_img = cv::Rect(box.x, box.y + roi_top, box.width, box.height);
  g_red_avoid.center_img = red_centroid_img;
  g_red_avoid.bottom_img = red_bottom_img;
  g_red_avoid.aspect_ratio = aspect_ratio;
  g_red_avoid.contour_area_ratio = contour_area_ratio;

  const double roi_h = std::max(1, roi_bottom - roi_top);
  g_red_avoid.bottom_ratio = clamp(
    (red_bottom_img.y - static_cast<double>(roi_top)) / roi_h,
    0.0,
    1.0);

  return true;
}

std::string LineFollowerNode::state_to_string(TrackState state) const
{
  switch (state) {
    case TrackState::TRACKING:
      return "tracking";
    case TrackState::YELLOW_LINE_DETECTED:
      return "yellow_line_detected";
    case TrackState::LOST:
    default:
      return "lost";
  }
}

void LineFollowerNode::publish_cmd_vel(
  double linear_speed, double yaw_rate, bool publish_stop, bool publish_avoid)
{
  geometry_msgs::msg::Twist cmd;
  cmd.linear.x = linear_speed;
  cmd.angular.z = yaw_rate;
  cmd_vel_pub_->publish(cmd);

  if (avoid_pub_ && publish_avoid && !publish_stop) {
    avoid_pub_->publish(cmd);
  }

  if (stop_pub_ && publish_stop && (linear_speed == 0.0 && yaw_rate == 0.0)) {
    stop_pub_->publish(cmd);
  }
}

void LineFollowerNode::publish_state(TrackState state)
{
  std_msgs::msg::String msg;
  msg.data = state_to_string(state);
  state_pub_->publish(msg);
}

void LineFollowerNode::publish_centroid(double norm_x, double norm_y)
{
  if (!centroid_pub_) {
    return;
  }
  geometry_msgs::msg::Point msg;
  msg.x = norm_x;
  msg.y = norm_y;
  centroid_pub_->publish(msg);
}

void LineFollowerNode::publish_debug_image(
  const sensor_msgs::msg::Image::ConstSharedPtr & source,
  const cv::Mat & frame_bgr,
  const ProcessingResult & result) const
{
  if (!debug_enabled_ || result.binary_mask.empty()) {
    return;
  }

  cv::Mat overlay = frame_bgr.clone();
  cv::Mat mask_color;
  cv::cvtColor(result.binary_mask, mask_color, cv::COLOR_GRAY2BGR);

  cv::Mat blended;
  cv::addWeighted(result.roi_bgr, 0.6, mask_color, 0.4, 0.0, blended);
  blended.copyTo(
    overlay(cv::Rect(0, result.roi_top, result.roi_bgr.cols, result.roi_bgr.rows)));

  if (result.line_detected) {
    if (result.bounding_box.width > 0 && result.bounding_box.height > 0) {
      cv::rectangle(
        overlay,
        cv::Rect(
          result.bounding_box.x,
          result.bounding_box.y + result.roi_top,
          result.bounding_box.width,
          result.bounding_box.height),
        cv::Scalar(0, 255, 0),
        2);
    }

    const cv::Point centroid(
      static_cast<int>(result.centroid.x),
      static_cast<int>(result.centroid.y) + result.roi_top);
    cv::circle(overlay, centroid, 8, cv::Scalar(0, 0, 255), -1);

    cv::line(
      overlay,
      cv::Point(overlay.cols / 2, result.roi_top),
      cv::Point(overlay.cols / 2, result.roi_bottom),
      cv::Scalar(0, 255, 255),
      2);

    cv::Point2d base(
      static_cast<double>(frame_bgr.cols) * 0.5,
      static_cast<double>(result.roi_bottom));
    cv::Point2d v_line(
      static_cast<double>(result.centroid.x) - base.x,
      static_cast<double>(result.centroid.y + result.roi_top) - base.y);

    if (red_divider_angle_rad_ != 0.0) {
      const double angle =
        v_line.x >= 0.0 ? red_divider_angle_rad_ : -red_divider_angle_rad_;
      const double c = std::cos(angle);
      const double s = std::sin(angle);
      v_line = cv::Point2d(
        v_line.x * c - v_line.y * s,
        v_line.x * s + v_line.y * c);
    }

    const cv::Point2d target = base + v_line;
    cv::line(overlay, base, target, cv::Scalar(0, 255, 255), 2);

    if (last_red_overlay_valid_ && g_red_avoid.valid) {
      cv::rectangle(overlay, g_red_avoid.bbox_img, cv::Scalar(0, 0, 255), 2);
      cv::circle(overlay, g_red_avoid.center_img, 6, cv::Scalar(0, 0, 255), -1);
      cv::circle(overlay, g_red_avoid.bottom_img, 6, cv::Scalar(0, 255, 255), -1);

      char red_text[200];
      std::snprintf(
        red_text,
        sizeof(red_text),
        "%s bottom=%.2f risk=%.2f area=%.4f %s%s",
        g_red_avoid.is_red_line ? "RED_ZONE" : "CONE",
        g_red_avoid.bottom_ratio,
        g_red_avoid.risk,
        g_red_avoid.contour_area_ratio,
        g_red_avoid.in_corridor ? "IN_PATH " : "OUT_PATH ",
        g_red_avoid.very_close ? "VERY_CLOSE" : "");

      cv::putText(
        overlay, red_text, cv::Point(10, 90),
        cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(0, 0, 255), 2);
    }

    if (g_red_avoid.red_line_detected) {
      cv::rectangle(overlay, g_red_avoid.red_line_bbox_img, cv::Scalar(255, 0, 255), 2);
      cv::putText(
        overlay, "RED ZONE AVOID",
        cv::Point(
          g_red_avoid.red_line_bbox_img.x,
          std::max(20, g_red_avoid.red_line_bbox_img.y - 5)),
        cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(255, 0, 255), 2);
    }
  }

  if (this->get_parameter("enable_continuous_curve_assist").as_bool()) {
    const char * curve_mode =
      tjunction_merging_ ? "T_MERGE" :
      (g_curve_assist.reverse_curve_active ? "REV_CURVE" : "NORMAL");
    char curve_text[180];
    std::snprintf(
      curve_text, sizeof(curve_text),
      "%s near=%+.2f far=%+.2f(w=%.2f) level=%.2f%s",
      curve_mode, g_curve_assist.near_error, g_curve_assist.far_error,
      g_curve_assist.far_weight, g_curve_assist.curve_level,
      tjunction_merge_armed_ ? " [T-ARMED]" : "");
    cv::putText(
      overlay, curve_text, cv::Point(10, 60),
      cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(255, 255, 0), 2);
  }

  if (result.yellow_line_detected) {
    cv::putText(
      overlay, "YELLOW LINE DETECTED!", cv::Point(10, 30),
      cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
  }

  cv_bridge::CvImage debug_image;
  debug_image.header = source->header;
  debug_image.encoding = "bgr8";
  debug_image.image = overlay;
  debug_pub_.publish(debug_image.toImageMsg());
}

double LineFollowerNode::clamp(double value, double min_value, double max_value)
{
  return std::max(min_value, std::min(value, max_value));
}

}  // namespace line_follower
