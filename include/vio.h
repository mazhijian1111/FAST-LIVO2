/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VIO_H_
#define VIO_H_

#include "voxel_map.h"
#include "feature.h"
#include <opencv2/imgproc/imgproc_c.h>
#include <pcl/filters/voxel_grid.h>
#include <set>
#include <vikit/math_utils.h>
#include <vikit/robust_cost.h>
#include <vikit/vision.h>
#include <vikit/pinhole_camera.h>

struct SubSparseMap
{
  vector<float> propa_errors;
  vector<float> errors;
  vector<vector<float>> warp_patch;
  vector<int> search_levels;
  vector<VisualPoint *> voxel_points;
  vector<double> inv_expo_list;
  vector<pointWithVar> add_from_voxel_map;

  SubSparseMap()
  {
    propa_errors.reserve(SIZE_LARGE);
    errors.reserve(SIZE_LARGE);
    warp_patch.reserve(SIZE_LARGE);
    search_levels.reserve(SIZE_LARGE);
    voxel_points.reserve(SIZE_LARGE);
    inv_expo_list.reserve(SIZE_LARGE);
    add_from_voxel_map.reserve(SIZE_SMALL);
  };

  void reset()
  {
    propa_errors.clear();
    errors.clear();
    warp_patch.clear();
    search_levels.clear();
    voxel_points.clear();
    inv_expo_list.clear();
    add_from_voxel_map.clear();
  }
};

class Warp
{
public:
  Matrix2d A_cur_ref;
  int search_level;
  Warp(int level, Matrix2d warp_matrix) : search_level(level), A_cur_ref(warp_matrix) {}
  ~Warp() {}
};

class VOXEL_POINTS
{
public:
  std::vector<VisualPoint *> voxel_points;
  int count;
  VOXEL_POINTS(int num) : count(num) {}
  ~VOXEL_POINTS() 
  { 
    for (VisualPoint* vp : voxel_points) 
    {
      if (vp != nullptr) { delete vp; vp = nullptr; }
    }
  }
};

class VIOManager
{
public:
  int grid_size;
  vk::AbstractCamera *cam;
  vk::PinholeCamera *pinhole_cam;
  StatesGroup *state;
  StatesGroup *state_propagat;
  M3D Rli, Rci, Rcl, Rcw, Jdphi_dR, Jdp_dt, Jdp_dR;
  V3D Pli, Pci, Pcl, Pcw;
  vector<int> grid_num;
  vector<int> map_index;
  vector<int> border_flag;
  vector<int> update_flag;
  vector<float> map_dist;
  vector<float> scan_value;
  vector<float> patch_buffer;
  bool normal_en, inverse_composition_en, exposure_estimate_en, raycast_en, has_ref_patch_cache;
  bool ncc_en = false, colmap_output_en = false;

  int width, height, grid_n_width, grid_n_height, length;
  double image_resize_factor;
  double fx, fy, cx, cy;
  int patch_pyrimid_level, patch_size, patch_size_total, patch_size_half, border, warp_len;
  int max_iterations, total_points;

  double img_point_cov, outlier_threshold, ncc_thre;
  // ---- Layer 7 (Theorem T7): robust t-distribution visual residual. ----
  // When robust_vio_enable is true, each photometric residual is reweighted
  // by a t-distribution weight w_j = (nu+2)/(nu + r_j^2/sigma^2), computed
  // from the running per-level residual variance sigma^2. This bounds the
  // influence of outlier pixels (long-baseline blur, lighting changes) so
  // the visual pseudo-information on degenerate directions is O(eps) instead
  // of O(1) — the breakdown bound of Theorem T7. When false, the original
  // fixed-sigma least-squares behaviour is unchanged.
  bool   robust_vio_enable = false;
  double robust_vio_nu    = 3.0;     // t-distribution DOF (3 = heavy tail)
  double robust_vio_sigma2 = 100.0; // running residual variance (EMA-updated)
  double robust_vio_alpha  = 0.1;   // EMA factor for sigma2 update
  // ---- Layer 5 (Theorem T5): cross-modal degeneracy-driven patch select. --
  // When enabled, the visual update selects a budget of top-k patches that
  // maximize the trace of their pose-information projected onto the LiDAR
  // degenerate subspace Π_deg (cached by VoxelMapManager::dd_last_probe_).
  // This quantitatively routes visual information into the directions where
  // LiDAR is weak (T5: a submodular greedy bound). When disabled, ALL valid
  // patches are used (original FAST-LIVO2 behaviour). The Π_deg matrix is
  // pushed in from LIVMapper (which owns both managers) each VIO frame.
  bool   cross_modal_select_enable = false;
  int    cross_modal_budget = -1;   // -1 = use all; >0 = top-k greedy
  Eigen::Matrix<double, 6, 6> lio_Pi_deg =
      Eigen::Matrix<double, 6, 6>::Zero();
  bool   lio_Pi_deg_valid = false;
  
  SubSparseMap *visual_submap;
  std::vector<std::vector<V3D>> rays_with_sample_points;

  double compute_jacobian_time, update_ekf_time;
  double ave_total = 0;
  // double ave_build_residual_time = 0;
  // double ave_ekf_time = 0;

  int frame_count = 0;
  bool plot_flag;

  Matrix<double, DIM_STATE, DIM_STATE> G, H_T_H;
  MatrixXd K, H_sub_inv;

  // ---- P2 fused-mask (Project A): cache the VIO pose information block ----
  // Λ_V (6×6 pose) + b_V (6×1 pose) from the most recent VIO ESIKF update, so
  // the LIO step can probe degeneracy on Λ_f = Λ_L + Λ_V (Theorem T1). The
  // cache is ONE VIO frame stale w.r.t. the current LIO step (LIO/VIO run at
  // different timestamps); slow-motion approximation, documented limitation.
  Eigen::Matrix<double, 6, 6> last_Lambda_V = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> last_b_V      = Eigen::Matrix<double, 6, 1>::Zero();
  bool last_V_valid = false;

  // ---- RR-IESKF Axis II (theory.tex sec:gate-optimal / prop:gate-optimal):
  // optimal per-modality reliability gate. Replaces the shipped consistency
  // ramp clip(1 - nu/kappa, 0, 1) by the cost-optimal hyperbola (G5):
  //   u* = min(1, 1 / (2 C b̂² λ_L − η)),   η = λ_L/(λ_V + π),
  // driven by the plug-in bias read b̂²_m = ⟨r_m²⟩ − 1/λ_m (prop:bias-read
  // B1/B2: its floor is the filter's own error variance — clamp at 0).
  // The gate multiplies the VISUAL pose information block by u² (the suspect
  // modality here is V; LiDAR is the reference). rem:common-meaning records
  // the domain: the gate covers per-modality degradation and is provably
  // inert to a steady common-mode (extrinsic) bias (prop:common-mode M5).
  bool   gate_enable   = false;   // master switch (vio/gate_enable)
  double gate_C        = 1.0;     // c_bias / c_var cost ratio (vio/gate_C)
  double gate_kappa    = 4.0;     // shipped-ramp threshold, kept for the log
  // EMA of the NIS proxy: mean photometric residual energy, in units of the
  // nominal per-pixel variance (img_point_cov). nu=1 ⇔ consistent (T2).
  double gate_nu_V     = 1.0;     // EMA of r^2 / img_point_cov (plug-in read + 1)
  double gate_nu_alpha = 0.05;    // EMA forgetting factor
  double gate_u_V      = 1.0;     // current gate factor on Λ_V (diagnostics)
  double gate_lambda_V = 1.0;     // scale of Λ_V: tr(Λ_V)/6, EMA-smoothed
  double gate_lambda_L = 1.0;     // scale of Λ_L from LIO: tr(Λ_L)/6 (set by LIVMapper)
  bool   gate_lambda_L_valid = false;
  // Per-direction differential bias read (prop:bias-read B3): the LiDAR-vs-
  // visual NIS difference, updated per frame by LIVMapper and consumed by
  // the per-direction gate. Blind to common-mode bias (prop:common-mode M2)
  // — the desired selectivity.
  Eigen::Matrix<double, 6, 1> gate_diff_read_ =
      Eigen::Matrix<double, 6, 1>::Zero();
  bool   gate_diff_valid = false;
  // Consistent gate state: when the per-direction gate modified Λ_V's pose
  // block (above), the SAME u_k weights must be applied to the residual
  // information vector b_V in Λ_V's eigenbasis (mean/covariance-consistent
  // weighted observation). Cached basis/u from the last gate application.
  bool   gate_b_V_scale_valid = false;
  Eigen::Matrix<double, 6, 6> gate_b_V_basis =
      Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 6, 1> gate_b_V_u =
      Eigen::Matrix<double, 6, 1>::Ones();

  ofstream fout_camera, fout_colmap;
  unordered_map<VOXEL_LOCATION, VOXEL_POINTS *> feat_map;
  unordered_map<VOXEL_LOCATION, int> sub_feat_map; 
  unordered_map<int, Warp *> warp_map;
  vector<VisualPoint *> retrieve_voxel_points;
  vector<pointWithVar> append_voxel_points;
  FramePtr new_frame_;
  cv::Mat img_cp, img_rgb, img_test;

  enum CellType
  {
    TYPE_MAP = 1,
    TYPE_POINTCLOUD,
    TYPE_UNKNOWN
  };

  VIOManager();
  ~VIOManager();
  void updateStateInverse(cv::Mat img, int level);
  void updateState(cv::Mat img, int level);
  void processFrame(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time);
  void retrieveFromVisualSparseMap(cv::Mat img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void generateVisualMapPoints(cv::Mat img, vector<pointWithVar> &pg);
  void setImuToLidarExtrinsic(const V3D &transl, const M3D &rot);
  void setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P);
  void initializeVIO();
  void getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level);
  void computeProjectionJacobian(V3D p, MD(2, 3) & J);
  void computeJacobianAndUpdateEKF(cv::Mat img);
  void resetGrid();
  void updateVisualMapPoints(cv::Mat img);
  void getWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref, const double depth_ref, const SE3 &T_cur_ref,
                           const int level_ref, 
                           const int pyramid_level, const int halfpatch_size, Matrix2d &A_cur_ref);
  void getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref,
                                     const V3D &xyz_ref, const V3D &normal_ref, const SE3 &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref);
  void warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                  const int pyramid_level, const int halfpatch_size, float *patch);
  void insertPointIntoVoxelMap(VisualPoint *pt_new);
  void plotTrackedPoints();
  void updateFrameState(StatesGroup state);
  void projectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void updateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void precomputeReferencePatches(int level);
  void dumpDataForColmap();
  double calculateNCC(float *ref_patch, float *cur_patch, int patch_size);
  int getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level);
  V3F getInterpolatedPixel(cv::Mat img, V2D pc);
  
  // void resetRvizDisplay();
  // deque<VisualPoint *> map_cur_frame;
  // deque<VisualPoint *> sub_map_ray;
  // deque<VisualPoint *> sub_map_ray_fov;
  // deque<VisualPoint *> visual_sub_map_cur;
  // deque<VisualPoint *> visual_converged_point;
  // std::vector<std::vector<V3D>> sample_points;

  // PointCloudXYZI::Ptr pg_down;
  // pcl::VoxelGrid<PointType> downSizeFilter;
};
typedef std::shared_ptr<VIOManager> VIOManagerPtr;

#endif // VIO_H_