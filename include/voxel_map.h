/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VOXEL_MAP_H_
#define VOXEL_MAP_H_

#include "common_lib.h"
#include "degeneracy.h"  // DD-ESIKF (Project A): direction-decoupled masking
#include <Eigen/Dense>
#include <fstream>
#include <fstream>
#include <math.h>
#include <mutex>
#include <omp.h>
#include <pcl/common/io.h>
#include <ros/ros.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#define VOXELMAP_HASH_P 116101
#define VOXELMAP_MAX_N 10000000000

static int voxel_plane_id = 0;

typedef struct VoxelMapConfig
{
  double max_voxel_size_;
  int max_layer_;
  int max_iterations_;
  std::vector<int> layer_init_num_;
  int max_points_num_;
  double planner_threshold_;
  double beam_err_;
  double dept_err_;
  double sigma_num_;
  bool is_pub_plane_map_;

  // config of local map sliding
  double sliding_thresh;
  bool map_sliding_en;
  int half_map_size;
} VoxelMapConfig;

typedef struct PointToPlane
{
  Eigen::Vector3d point_b_;
  Eigen::Vector3d point_w_;
  Eigen::Vector3d normal_;
  Eigen::Vector3d center_;
  Eigen::Matrix<double, 6, 6> plane_var_;
  M3D body_cov_;
  int layer_;
  double d_;
  double eigen_value_;
  bool is_valid_;
  float dis_to_plane_;
} PointToPlane;

typedef struct VoxelPlane
{
  Eigen::Vector3d center_;
  Eigen::Vector3d normal_;
  Eigen::Vector3d y_normal_;
  Eigen::Vector3d x_normal_;
  Eigen::Matrix3d covariance_;
  Eigen::Matrix<double, 6, 6> plane_var_;
  float radius_ = 0;
  float min_eigen_value_ = 1;
  float mid_eigen_value_ = 1;
  float max_eigen_value_ = 1;
  float d_ = 0;
  int points_size_ = 0;
  bool is_plane_ = false;
  bool is_init_ = false;
  int id_ = 0;
  bool is_update_ = false;
  VoxelPlane()
  {
    plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
    covariance_ = Eigen::Matrix3d::Zero();
    center_ = Eigen::Vector3d::Zero();
    normal_ = Eigen::Vector3d::Zero();
  }
} VoxelPlane;

class VOXEL_LOCATION
{
public:
  int64_t x, y, z;

  VOXEL_LOCATION(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOCATION &other) const { return (x == other.x && y == other.y && z == other.z); }
};

// Hash value
namespace std
{
template <> struct hash<VOXEL_LOCATION>
{
  int64_t operator()(const VOXEL_LOCATION &s) const
  {
    using std::hash;
    using std::size_t;
    return ((((s.z) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.y)) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.x);
  }
};
} // namespace std

struct DS_POINT
{
  float xyz[3];
  float intensity;
  int count = 0;
};

void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov);

class VoxelOctoTree
{

public:
  VoxelOctoTree() = default;
  std::vector<pointWithVar> temp_points_;
  VoxelPlane *plane_ptr_;
  int layer_;
  int octo_state_; // 0 is end of tree, 1 is not
  VoxelOctoTree *leaves_[8];
  double voxel_center_[3]; // x, y, z
  std::vector<int> layer_init_num_;
  float quater_length_;
  float planer_threshold_;
  int points_size_threshold_;
  int update_size_threshold_;
  int max_points_num_;
  int max_layer_;
  int new_points_;
  bool init_octo_;
  bool update_enable_;

  VoxelOctoTree(int max_layer, int layer, int points_size_threshold, int max_points_num, float planer_threshold)
      : max_layer_(max_layer), layer_(layer), points_size_threshold_(points_size_threshold), max_points_num_(max_points_num),
        planer_threshold_(planer_threshold)
  {
    temp_points_.clear();
    octo_state_ = 0;
    new_points_ = 0;
    update_size_threshold_ = 5;
    init_octo_ = false;
    update_enable_ = true;
    for (int i = 0; i < 8; i++)
    {
      leaves_[i] = nullptr;
    }
    plane_ptr_ = new VoxelPlane;
  }

  ~VoxelOctoTree()
  {
    for (int i = 0; i < 8; i++)
    {
      delete leaves_[i];
    }
    delete plane_ptr_;
  }
  void init_plane(const std::vector<pointWithVar> &points, VoxelPlane *plane);
  void init_octo_tree();
  void cut_octo_tree();
  void UpdateOctoTree(const pointWithVar &pv);

  VoxelOctoTree *find_correspond(Eigen::Vector3d pw);
  VoxelOctoTree *Insert(const pointWithVar &pv);
};

void loadVoxelConfig(ros::NodeHandle &nh, VoxelMapConfig &voxel_config);

class VoxelMapManager
{
public:
  VoxelMapManager() = default;
  VoxelMapConfig config_setting_;
  int current_frame_id_ = 0;
  ros::Publisher voxel_map_pub_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map_;

  PointCloudXYZI::Ptr feats_undistort_;
  PointCloudXYZI::Ptr feats_down_body_;
  PointCloudXYZI::Ptr feats_down_world_;

  M3D extR_;
  V3D extT_;
  float build_residual_time, ekf_time;
  float ave_build_residual_time = 0.0;
  float ave_ekf_time = 0.0;
  int scan_count = 0;
  StatesGroup state_;
  V3D position_last_;

  V3D last_slide_position = {0,0,0};

  geometry_msgs::Quaternion geoQuat_;

  int feats_down_size_;
  int effct_feat_num_;
  std::vector<M3D> cross_mat_list_;
  std::vector<M3D> body_cov_list_;
  std::vector<pointWithVar> pv_list_;
  std::vector<PointToPlane> ptpl_list_;

  VoxelMapManager(VoxelMapConfig &config_setting, std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &voxel_map)
      : config_setting_(config_setting), voxel_map_(voxel_map)
  {
    current_frame_id_ = 0;
    feats_undistort_.reset(new PointCloudXYZI());
    feats_down_body_.reset(new PointCloudXYZI());
    feats_down_world_.reset(new PointCloudXYZI());
  };

  void StateEstimation(StatesGroup &state_propagat);
  void TransformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud);

  void BuildVoxelMap();
  V3F RGBFromVoxel(const V3D &input_point);

  void UpdateVoxelMap(const std::vector<pointWithVar> &input_points);

  void BuildResidualListOMP(std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list);

  void build_single_residual(pointWithVar &pv, const VoxelOctoTree *current_octo, const int current_layer, bool &is_sucess, double &prob,
                             PointToPlane &single_ptpl);

  void pubVoxelMap();

  void mapSliding();
  void clearMemOutOfMap(const int& x_max,const int& x_min,const int& y_max,const int& y_min,const int& z_max,const int& z_min );

  // ---- DD-ESIKF (Project A) -------------------------------------------
  // Degeneracy-resilient fusion: optional per-frame masking + prior injection
  // on unobservable directions of the LiDAR information matrix Λ_L.
  // Enable via config: `degeneracy_enable` (default false => original ESIKF).
  dd_esikf::DegeneracyConfig dd_cfg_;
  std::shared_ptr<dd_esikf::PriorSource> dd_prior_src_;  // IMU/visual prior (optional)
  bool dd_enable_ = false;                              // runtime toggle
  bool dd_fused_mask_ = true;  // T1: probe on Λ_f = Λ_L + Λ_V (gates Λ_V use)
  // Latest probe result, exposed for diagnostics / plotting.
  dd_esikf::DegeneracyResult dd_last_probe_;
  // Per-frame cache of effective Λ and H^T R^-1 z so inner ESIKF iterations
  // reuse the same masked quantities (Assumption 1 / Proposition 5).
  Eigen::Matrix<double, 6, 6> dd_Lambda_eff_cached_ = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> dd_Htz_eff_cached_    = Eigen::Matrix<double, 6, 1>::Zero();
  // ---- P2 fused-mask (Theorem T1): visual information from the last VIO
  // frame, set by LIVMapper before calling StateEstimation. When left zero,
  // the degeneracy probe falls back to Λ_L-only (original DD-ESIKF). ----
  Eigen::Matrix<double, 6, 6> dd_Lambda_V_ = Eigen::Matrix<double, 6, 6>::Zero();
  bool dd_Lambda_V_valid_ = false;
  // ---- Layer 2 (Theorem T2): NIS-based adaptive LiDAR measurement noise. ----
  dd_esikf::AdaptiveNoiseConfig dd_ada_cfg_;
  dd_esikf::AdaptiveNoiseState  dd_ada_state_;
  double dd_ada_phi_ = 1.0;   // current R scaling factor for LiDAR
  // ---- Layer 6 (Theorem T6): range-dependent anisotropic LiDAR R. ----
  dd_esikf::AnisoNoiseConfig dd_aniso_cfg_;
  // ---- Layer 4 (Theorem T4): first-estimate Jacobian (FEJ) mode. ----
  // When fej_enable is true, the pose-block information matrix Λ_L, the
  // observation-info vector HTz, AND the prior term (state_propagat - state_)
  // used in the ESIKF update are all frozen at the FIRST inner iteration's
  // linearization point and held fixed across inner iterations. This removes
  // the fictitious-observability leak (F5): a re-linearizing ESIKF lets small
  // eigenvalues drift as H shifts iteration to iteration, injecting spurious
  // information into unobservable directions. FEJ holds the linearization
  // point fixed so the information added per direction is consistent with the
  // geometry at the first estimate. Cached in fej_*_cached_ at iterCount==0.
  bool fej_enable_ = false;
  Eigen::Matrix<double, 6, 6> dd_Lambda_L_cached_ = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> dd_Htz_cached_      = Eigen::Matrix<double, 6, 1>::Zero();
  // ---- RR-IESKF Axis I (theory.tex eq:dfej / lem:dfej): directional FEJ.
  // Instead of freezing the WHOLE pose information block (fej_enable_), DFEJ
  // freezes it ONLY on the degenerate subspace and re-linearizes on the
  // observable complement each inner iteration:
  //   Λ_eff   = Π_obs Λ^(i) Π_obs + Π_deg Λ^(0)_eff Π_deg
  //   HTz_eff = Π_obs HTz^(i)   + Π_deg HTz^(0)_eff
  // (Λ^(0)_eff = the DD-masked effective info at the first iterate when DD is
  // on, else the raw first-estimate Λ_L.) The projectors Π_obs/Π_deg come from
  // dd_last_probe_ (computed at iterCount==0), so DFEJ requires dd_enable_.
  // The O→D cross-covariance leak (rem:dfej-leak) is blocked by the thm:main
  // P4 covariance projection, which runs later in the same update. ----
  bool dfej_enable_ = false;
  // Scale-relative plane acceptance (thm:residual / cor:tilt-resolution).
  // VoxelOctoTree::init_plane is a free-standing class without access to the
  // manager's config, so the tilt threshold is relayed through a static
  // set once at init (initDegeneracy). 0.0 => legacy absolute
  // planer_threshold_; >0 => λ_min/λ_mid < tilt_tau².
  double tilt_tau_ = 0.0;
  static double & s_tilt_tau()
  {
    static double v = 0.0;
    return v;
  }
  static double tilt_tau_global() { return s_tilt_tau(); }
  // RR-IESKF external anchor (prop:common-mode M4 / eq:anchor-att).
  // Full-rank pseudo-observation with information anchor_lambda_ on ALL pose
  // directions; attenuates a steady common-mode bias by
  // (λ_L+λ_V)/(λ_L+λ_V+λ_A). anchor_obs_ is the anchor residual (pose-error
  // units, same value on all 6 dims by default — a differential anchor would
  // fill the 6-vector per frame); z_A = 0 means pull toward the propagated
  // pose.
  bool   anchor_enable_ = false;
  double anchor_lambda_ = 0.0;
  double anchor_obs_    = 0.0;
  Eigen::Matrix<double, 6, 1> anchor_z_A_ = Eigen::Matrix<double, 6, 1>::Zero();
  // Real IMU preintegration prior (T2/T3): set in initDegeneracy, wired by
  // LIVMapper each LIO frame. When false, the B3 constant floor
  // (default_prior_info) is the only degenerate-subspace prior.
  bool imu_prior_enable_ = false;
  // Optional file logger for per-frame degeneracy probe (Project A eval).
  std::string dd_log_file_;                          // empty => disabled
  std::ofstream dd_log_;                             // opened in initDegeneracy
  int dd_frame_idx_ = 0;                              // monotonic frame counter
  void initDegeneracy(ros::NodeHandle &nh);              // read params from ROS
  // P2: receive the cached VIO pose information block (called by LIVMapper).
  void setVisualInfo(const Eigen::Matrix<double, 6, 6>& Lambda_V,
                     bool valid) {
    dd_Lambda_V_ = Lambda_V;
    dd_Lambda_V_valid_ = valid;
  }

private:
  void GetUpdatePlane(const VoxelOctoTree *current_octo, const int pub_max_voxel_layer, std::vector<VoxelPlane> &plane_list);

  void pubSinglePlane(visualization_msgs::MarkerArray &plane_pub, const std::string plane_ns, const VoxelPlane &single_plane, const float alpha,
                      const Eigen::Vector3d rgb);
  void CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec, geometry_msgs::Quaternion &q);

  void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b);
};
typedef std::shared_ptr<VoxelMapManager> VoxelMapManagerPtr;

#endif // VOXEL_MAP_H_
