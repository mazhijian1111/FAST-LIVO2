/*
 * degeneracy.h — Direction-Decoupled ESIKF (DD-ESIKF) for FAST-LIVO2.
 *
 * Project A: Selective degeneracy-resilient fusion. Implements the online
 * spectral decomposition of the LiDAR information matrix Λ_L, identification
 * of the degenerate (unobservable / weakly-observable) subspace, construction
 * of a continuous direction mask M, and injection of an unbiased prior
 * (IMU / visual / kinematic) on the degenerate directions only.
 *
 * Theory: see research/theory.tex (Theorems 1–4, Proposition 5).
 * Simulation: see research/ddesikf_sim.py (all six theorem-checks PASS).
 *
 * This header is header-only and depends only on Eigen + common_lib.h types.
 * It is designed to drop into VoxelMapManager::StateEstimation at
 * voxel_map.cpp:466 with a ~30-line modification (see ApplyToESIKF below).
 *
 * Author: Project A (DD-ESIKF). License: BSD (same as FAST-LIVO2).
 */
#ifndef DEGENERACY_H_
#define DEGENERACY_H_

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>
#ifdef DEGENERACY_STANDALONE_TEST
  // Standalone compile-test path: common_lib.h is stubbed by the test TU.
# include <sophus/so3.h>
using namespace Sophus;  // NOLINT
using V3D = Eigen::Vector3d;
using M3D = Eigen::Matrix3d;
#else
# include "common_lib.h"  // V3D, M3D, MD, Sophus SO3 (via `using namespace Sophus`)
#endif

// Only the 6-DOF pose error [δθ(3), δp(3)] is coupled to LiDAR point-to-plane
// residuals; the degeneracy analysis lives in this 6-dim subspace.
static constexpr int kPoseDim = 6;

namespace dd_esikf {

// ============================================================================
// 1. Configuration
// ============================================================================
struct DegeneracyConfig {
  // Dual degeneracy criterion (Definition 1 in theory.tex).
  // A direction v_k is degenerate iff  λ_k < tau_abs  OR  λ_k/λ_1 < tau_rel.
  double tau_abs = 1.0;   // absolute threshold on λ_k (scene/sensor tuned)
  double tau_rel = 1e-3;  // relative threshold vs. the largest eigenvalue

  // Mask shape: m_k = clip(λ_k / tau_abs, 0, 1). A direction with λ_k = 0
  // gets m_k = 0 (fully masked); a direction with λ_k = tau_abs gets m_k = 1
  // (unmasked); in between the mask is continuous, avoiding hard-threshold
  // chatter (Proposition 5 / Assumption 1 require iterate-independence, and a
  // continuous mask also keeps Lambda_eff PSD).
  //
  // Optional: hard-mask the weakly-degenerate directions for a stricter
  // consistency guarantee (Corollary), at the cost of rejecting some genuine
  // weak information. Default false (continuous).
  bool hard_mask = false;

  // Minimum eigenvalue floor injected along the degenerate subspace when a
  // prior source is attached. This is λ_{pr,k} in Theorem 3. Set to 0 to
  // disable prior injection (pure masking, "B2" in the experiment grid).
  double default_prior_info = 0.0;

  // If true, the mask and prior are computed ONCE per frame (before the ESIKF
  // inner iteration loop) and held constant across inner iterations.
  // Required by Assumption 1 (iterate-independence) for Proposition 5
  // (convergence). KEEP THIS TRUE unless you have a reason to re-linearize.
  bool per_frame_mask = true;
};

// ============================================================================
// 2. Degeneracy probe (Theorem 1 / Definition 1)
// ============================================================================
struct DegeneracyResult {
  // Spectral decomposition of Λ_L (6×6), eigenvalues in DESCENDING order.
  Eigen::Matrix<double, kPoseDim, 1> eigvals;   // λ_1 ≥ ... ≥ λ_6 ≥ 0
  Eigen::Matrix<double, kPoseDim, kPoseDim> eigvecs;  // columns = v_k
  Eigen::Array<bool, kPoseDim, 1> deg_mask;     // true => degenerate direction

  // Continuous mask M = Σ m_k v_k v_k^T  (Theorem 3).
  Eigen::Matrix<double, kPoseDim, kPoseDim> M;
  Eigen::Matrix<double, kPoseDim, 1> m_coefs;   // the m_k values

  // Indices (into the descending-order eigvecs columns) of degenerate dirs.
  std::vector<int> deg_indices;
  std::vector<int> obs_indices;

  // Convenience: the weakly-degenerate direction (largest λ_k among deg)
  // and the strictly-unobservable direction (smallest λ_k). May be the same
  // if only one degenerate direction exists.
  int idx_weak = -1;
  int idx_strict = -1;

  bool has_degeneracy() const { return !deg_indices.empty(); }
  int deg_rank() const { return deg_indices.size(); }

  // ---- Layer 3 (Theorem T3): observable / degenerate projectors on the
  // 6-dim pose subspace.  Π_obs = Σ_{k∈obs} v_k v_k^T,  Π_deg = I − Π_obs.
  // Used by the direction-decoupled covariance update in voxel_map.cpp so
  // the degenerate-direction variance is NOT falsely contracted by the
  // (I − GH) factor — it keeps the prior (IMU-propagated) covariance, which
  // grows with distance per the IMU process noise rather than collapsing
  // to a fictitiously small value. On V_obs the projectors are identity and
  // the update reduces to the standard EKF posterior (T3(iii)). ----
  Eigen::Matrix<double, kPoseDim, kPoseDim> Pi_obs =
      Eigen::Matrix<double, kPoseDim, kPoseDim>::Identity();
  Eigen::Matrix<double, kPoseDim, kPoseDim> Pi_deg =
      Eigen::Matrix<double, kPoseDim, kPoseDim>::Zero();
};

// Compute the eigendecomposition of Λ_L and assemble the mask. Λ_L must be
// symmetric PSD (it is H^T R^-1 H, so this holds). Cost: one 6×6 SelfAdjoint
// eigen-decomposition per frame — microseconds.
inline DegeneracyResult probe(const Eigen::Matrix<double, kPoseDim, kPoseDim>& Lambda_L,
                              const DegeneracyConfig& cfg) {
  DegeneracyResult r;
  // SelfAdjointEigenSolver returns ASCENDING eigenvalues; we want DESCENDING
  // to match the theory's convention λ_1 ≥ ... ≥ λ_6.
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, kPoseDim, kPoseDim>> es(
      Lambda_L);
  Eigen::Matrix<double, kPoseDim, 1> evals = es.eigenvalues();        // ascending
  Eigen::Matrix<double, kPoseDim, kPoseDim> evecs = es.eigenvectors();  // cols

  // Reverse to descending order.
  for (int i = 0; i < kPoseDim; ++i) {
    r.eigvals(i) = evals(kPoseDim - 1 - i);
    r.eigvecs.col(i) = evecs.col(kPoseDim - 1 - i);
  }

  const double lam1 = r.eigvals(0);
  r.deg_mask.resize(kPoseDim);
  r.m_coefs.setZero();
  r.M.setZero();
  for (int k = 0; k < kPoseDim; ++k) {
    const double lk = r.eigvals(k);
    const bool deg = (lk < cfg.tau_abs) || (lk / std::max(lam1, 1e-12) < cfg.tau_rel);
    r.deg_mask(k) = deg;
    // Continuous mask: m_k = clip(λ_k / tau_abs, 0, 1).
    double mk = cfg.hard_mask ? (deg ? 0.0 : 1.0)
                              : std::max(0.0, std::min(1.0, lk / cfg.tau_abs));
    r.m_coefs(k) = mk;
    r.M += mk * (r.eigvecs.col(k) * r.eigvecs.col(k).transpose());
    if (deg) r.deg_indices.push_back(k);
    else     r.obs_indices.push_back(k);
  }
  if (!r.deg_indices.empty()) {
    r.idx_weak   = r.deg_indices.front();   // largest λ_k among degenerate
    r.idx_strict = r.deg_indices.back();    // smallest λ_k
  }
  // Layer 3 projectors: Π_obs over observable, Π_deg = I − Π_obs.
  // With a continuous mask the boundary between obs/deg is soft, but the
  // covariance projection uses the HARD degenerate set (the corollary
  // regime) so that strictly/weakly-unobservable variance is never
  // contracted by the LiDAR residual.
  r.Pi_obs.setZero();
  for (int k : r.obs_indices)
    r.Pi_obs += r.eigvecs.col(k) * r.eigvecs.col(k).transpose();
  r.Pi_deg = Eigen::Matrix<double, kPoseDim, kPoseDim>::Identity() - r.Pi_obs;
  return r;
}

// ============================================================================
// 3. Prior injection (Theorem 2 / Theorem 3)
// ============================================================================
// A prior source provides, for each degenerate direction, an information
// λ_{pr,k} and an unbiased observation z_{pr,k} (e.g. an IMU-preintegration
// displacement projected onto v_k). Subclass / specialize for IMU / visual /
// kinematic priors.
struct PriorObservation {
  bool valid = false;
  // Information along v_k (≥0). Larger => stronger pull toward z_pr.
  double info = 0.0;
  // Unbiased observation along v_k (state-space units). For an IMU
  // displacement prior, this is the v_k-projection of the IMU Δpose.
  double observation = 0.0;
};

class PriorSource {
 public:
  virtual ~PriorSource() = default;
  // Return a prior observation for degenerate direction v_k (column k of
  // res.eigvecs). Called once per frame, AFTER the probe.
  virtual PriorObservation priorForDirection(int k,
                                             const DegeneracyResult& res) const {
    (void)k; (void)res;
    return PriorObservation{};  // default: no prior
  }
};

// Assemble the prior information matrix Λ_pr (6×6) and prior observation
// vector b_pr (6,) from a prior source, restricted to the degenerate
// subspace. Non-degenerate directions get zero prior (do not pollute LiDAR).
inline void assemblePrior(const PriorSource* src,
                          const DegeneracyResult& res,
                          const DegeneracyConfig& cfg,
                          Eigen::Matrix<double, kPoseDim, kPoseDim>& Lambda_pr,
                          Eigen::Matrix<double, kPoseDim, 1>& b_pr) {
  Lambda_pr.setZero();
  b_pr.setZero();
  if (!src || !res.has_degeneracy()) return;
  for (int k : res.deg_indices) {
    const PriorObservation po = src->priorForDirection(k, res);
    if (!po.valid || po.info <= 0.0) continue;
    const auto v_k = res.eigvecs.col(k);
    Lambda_pr += po.info * (v_k * v_k.transpose());
    b_pr += po.info * po.observation * v_k;
  }
  // Fallback default floor on degenerate directions if the source gave nothing.
  if (cfg.default_prior_info > 0.0 && Lambda_pr.isZero()) {
    for (int k : res.deg_indices) {
      const auto v_k = res.eigvecs.col(k);
      Lambda_pr += cfg.default_prior_info * (v_k * v_k.transpose());
    }
  }
}

// ============================================================================
// 4. ApplyToESIKF — the three-line replacement at voxel_map.cpp:466
// ============================================================================
// Given the LiDAR information matrix Λ_L and the LiDAR observation-info
// vector Htz = H^T R^-1 z (both 6-dim), plus the mask and prior, produce the
// EFFECTIVE quantities Λ_eff and Htz_eff that the rest of the ESIKF update
// (voxel_map.cpp:468-490) should use:
//
//   Λ_eff   = M Λ_L M + Λ_pr          (Theorem 3)
//   Htz_eff = M Htz + b_pr            (Theorem 3)
//
// Then in the ESIKF loop:
//   K1 = (Λ_eff + P^-1)^{-1}
//   G  = K1 Λ_eff
//   δξ = K1 Htz_eff + (I - G)(x_prop - x)
//   P  = (I - G) P
//
// All three lines are the SAME as the original code, just with Λ_L→Λ_eff
// and Htz→Htz_eff substituted. No filter rewrite (Theorem 3).
//
// --- P2 extension (fused-mask, Theorem T1) -------------------------------
// FAST-LIVO2 runs LIO and VIO as two DECOUPLED sequential ESIKF updates on the
// shared state (no joint H_T_H exists). We therefore CANNOT add Λ_V into the
// LIO Kalman gain directly. Instead:
//   * probe() is run on the FUSED information  Λ_f = Λ_L + Λ_V  to obtain the
//     degenerate subspace V_deg (Theorem T1(iii): V_deg(Λ_f) ⊆ V_deg(Λ_L),
//     so visual info can only SHRINK the degenerate set → fewer prior
//     injections, less reliance on the prior).
//   * the mask M (built from Λ_f's spectrum) is applied to Λ_L ONLY, because
//     only the LIO update is being modified here; the visual ESIKF keeps its
//     own full Λ_V. Theorem T1(i): on V_obs, M≈I so Λ_L is untouched and LiDAR
//     drives the well-observed axes normally.
//   * Λ_V is the pose-block (rows/cols 0..5) of the VIO H_T_H from the most
//     recent VIO frame — one frame stale w.r.t. the current LIO step (the two
//     run at different timestamps). Under slow motion this is a good
//     approximation; document it as a limitation.
struct ESIKFInputs {
  Eigen::Matrix<double, kPoseDim, kPoseDim> Lambda_L;  // H_L^T R_L^-1 H_L
  Eigen::Matrix<double, kPoseDim, 1> Htz;               // H_L^T R_L^-1 z_L
  // Fused-mask (P2): visual pose information from the last VIO frame.
  // Zero (default) => probe degenerates to the original Λ_L-only behaviour.
  Eigen::Matrix<double, kPoseDim, kPoseDim> Lambda_V =
      Eigen::Matrix<double, kPoseDim, kPoseDim>::Zero();
};
struct ESIKFOutputs {
  Eigen::Matrix<double, kPoseDim, kPoseDim> Lambda_eff;
  Eigen::Matrix<double, kPoseDim, 1> Htz_eff;
  DegeneracyResult probe_res;  // for logging/diagnostics
};

inline ESIKFOutputs applyToESIKF(const ESIKFInputs& in,
                                 const DegeneracyConfig& cfg,
                                 const PriorSource* prior_src = nullptr) {
  ESIKFOutputs out;
  // P2 fused mask: probe on Λ_f = Λ_L + Λ_V so the degenerate subspace is
  // measured AFTER visual information is counted (Theorem T1(iii)). When
  // Λ_V==0 this reduces exactly to the original Λ_L-only probe.
  Eigen::Matrix<double, kPoseDim, kPoseDim> Lambda_f =
      in.Lambda_L + in.Lambda_V;
  out.probe_res = probe(Lambda_f, cfg);

  Eigen::Matrix<double, kPoseDim, kPoseDim> Lambda_pr;
  Eigen::Matrix<double, kPoseDim, 1> b_pr;
  assemblePrior(prior_src, out.probe_res, cfg, Lambda_pr, b_pr);

  // Theorem 3 + T1: mask is applied to Λ_L only (visual ESIKF is separate).
  // On V_obs, M≈I => Λ_eff|V_obs = Λ_L|V_obs (LiDAR untouched, T1(i)).
  // On V_deg, M≈0 => Λ_eff|V_deg = Λ_pr (prior only, T1(ii)).
  out.Lambda_eff = out.probe_res.M * in.Lambda_L * out.probe_res.M + Lambda_pr;
  out.Htz_eff    = out.probe_res.M * in.Htz + b_pr;
  return out;
}

// ============================================================================
// 5. Optional: IMU preintegration prior (example concrete PriorSource)
// ============================================================================
// Projects the IMU-propagated Δpose onto each degenerate direction and uses
// it as an unbiased prior observation. The information λ_{pr,k} should be
// set from the IMU preintegration covariance (inverse of the v_k-projection
// of the IMU Δpose covariance). Here we expose a simple version; a fuller
// implementation would pull Σ_Δpose from ImuProcess.
class IMUPreintegrationPrior : public PriorSource {
 public:
  IMUPreintegrationPrior(const V3D& dpos_imu,
                         const M3D& drot_imu,
                         const Eigen::Matrix<double, 6, 6>& imu_delta_cov,
                         double info_scale = 1.0)
      : dpos_(dpos_imu),
        drot_(drot_imu),
        cov_(imu_delta_cov),
        info_scale_(info_scale) {}

  // For degenerate direction v_k = [v_θ ; v_p] (rotation block | position block),
  // the prior observation is the v_k-projection of [log(ΔR) ; Δp].
  PriorObservation priorForDirection(int k,
                                     const DegeneracyResult& res) const override {
    if (k < 0 || k >= kPoseDim) return PriorObservation{};
    const auto v_k = res.eigvecs.col(k);
    Eigen::Matrix<double, 6, 1> delta;
    delta.head<3>() = SO3(drot_).log();  // small Δθ (old Sophus: SO3, not SO3d)
    delta.tail<3>() = dpos_;
    const double obs = v_k.dot(delta);
    // Information = inverse of the v_k-projection of the IMU Δpose covariance.
    const double var_k = v_k.dot(cov_ * v_k);
    const double info = (var_k > 1e-9) ? info_scale_ / var_k : 0.0;
    return PriorObservation{true, info, obs};
  }

 private:
  V3D dpos_;
  M3D drot_;
  Eigen::Matrix<double, 6, 6> cov_;
  double info_scale_;
};

// ============================================================================
// 6. Diagnostics
// ============================================================================
#ifndef DEGENERACY_STANDALONE_TEST
inline void logDegeneracy(const DegeneracyResult& r, ros::NodeHandle& /*nh*/) {
  // Cheap stdout diagnostic; can be wired to a ROS topic if desired.
  printf("[DD-ESIKF] Lambda_f eigvals: ");
  for (int k = 0; k < kPoseDim; ++k) printf("%.3e ", r.eigvals(k));
  printf("\n[DD-ESIKF] mask m_k: ");
  for (int k = 0; k < kPoseDim; ++k) printf("%.3f ", r.m_coefs(k));
  printf("\n[DD-ESIKF] degenerate dirs: %d  (weak=%.3e, strict=%.3e)\n",
         r.deg_rank(),
         r.idx_weak >= 0 ? r.eigvals(r.idx_weak) : -1.0,
         r.idx_strict >= 0 ? r.eigvals(r.idx_strict) : -1.0);
}
#else
inline void logDegeneracy(const DegeneracyResult& r) {
  printf("[DD-ESIKF] eigvals: ");
  for (int k = 0; k < kPoseDim; ++k) printf("%.3e ", r.eigvals(k));
  printf("\n[DD-ESIKF] mask m_k: ");
  for (int k = 0; k < kPoseDim; ++k) printf("%.3f ", r.m_coefs(k));
  printf("\n[DD-ESIKF] deg_rank: %d\n", r.deg_rank());
}
#endif

// ============================================================================
// 7. Adaptive measurement noise (Layer 2 / Theorem T2)
// ============================================================================
// NIS (normalized innovation squared) based per-modality adaptive R scaling.
// Maintains an EMA rho of the NIS residual statistic; in healthy (observable)
// operation rho -> 1 and phi -> 1 (no intervention). Under a degenerate /
// failing modality the NIS grows, rho grows, phi grows, and R is scaled up
// (=> H^T R^-1 H shrinks => the modality contributes less information,
// doubly suppressing it together with the directional mask of Layer 1).
//
//   rho_k = (1-alpha) rho_{k-1} + alpha * nu_k,   nu_k = r^T S^-1 r  (NIS)
//   phi_k = clip(phi_min + (phi_max - phi_min) * f(rho), phi_min, phi_max)
//   R_eff = R_nom * phi_k        (or equivalently R_inv_eff = R_inv_nom / phi_k)
//
// f is a saturating ramp; f(1)=0 so a healthy modality is left untouched.
// This is the Sage-Husa idea hardened to be a NO-OP when the filter is
// consistent (Theorem T2): on V_obs, nu ~ chi2(d), rho -> 1, phi -> 1.
struct AdaptiveNoiseConfig {
  bool   enable   = false;
  double alpha    = 0.05;   // EMA forgetting factor (small => slow adaptation)
  double phi_min  = 1.0;    // floor on the R scaling (never shrink R below nom)
  double phi_max  = 20.0;   // ceiling (avoid unbounded R inflation)
  double ramp_thr = 1.5;   // rho above which f starts ramping up
};
struct AdaptiveNoiseState {
  double rho = 1.0;       // EMA of NIS
  double phi = 1.0;       // current R scaling factor
};

// ============================================================================
// 8. Layer 6: range-dependent anisotropic LiDAR measurement noise (T6)
// ============================================================================
// For long-range points the depth (beam) variance grows ~range^2 but is
// largely discarded when the per-point covariance is projected onto the
// plane normal. We recover distance information by inflating R (shrinking
// R_inv) for far points, so the degenerate depth direction is down-weighted
// and degeneracy detection is sharper. Saturates at far_scale. No-op when
// enable is false (aniso_scale = 1).
struct AnisoNoiseConfig {
  bool   enable     = false;
  double far_range  = 30.0;   // range [m] beyond which R starts inflating
  double far_scale  = 5.0;    // max multiplicative inflation of R
};
inline double updateAdaptiveNoise(AdaptiveNoiseState& s,
                                  double nis,
                                  const AdaptiveNoiseConfig& cfg) {
  if (!cfg.enable) { s.rho = 1.0; s.phi = 1.0; return 1.0; }
  s.rho = (1.0 - cfg.alpha) * s.rho + cfg.alpha * nis;
  // Saturating ramp: 0 below ramp_thr, then linear to 1 at a large rho.
  double f = 0.0;
  if (s.rho > cfg.ramp_thr) {
    f = (s.rho - cfg.ramp_thr) / (s.rho + 1e-9);  // bounded in [0,1)
  }
  s.phi = cfg.phi_min + (cfg.phi_max - cfg.phi_min) * f;
  return s.phi;
}

// ============================================================================
// 9. RR-IESKF Axis II helpers (theory.tex sec:gate-optimal / sec:bias-read /
//    sec:common-mode)
// ============================================================================
namespace rr_ieskf {

// --- (a) Per-direction optimal gate (prop:gate-optimal, eq:gate-optimal) ---
// The theory's gate is PER DIRECTION: w_{m,k}^2 λ_{m,k} with u_k^* =
// min(1, 1/(2C b̂_k² λ_{m,k} − η_k)), η_k = λ_{m,k}/(λ_{m̄,k}+π_k). A scalar
// gate on the whole Λ block over-suppresses healthy directions in mixed
// scenes. This helper applies u_k per eigendirection of Λ_m:
//   Λ_gated = Σ_k u_k² λ_k v_k v_k^T
// (u_k=1 for all k reduces exactly to Λ_m). Cost: one 6×6 eigendecomposition.
struct GateInputs {
  Eigen::Matrix<double, kPoseDim, kPoseDim> Lambda;      // modality info Λ_m
  Eigen::Matrix<double, kPoseDim, 1> u;                  // per-direction gates
};
inline Eigen::Matrix<double, kPoseDim, kPoseDim> gatePerDirection(
    const GateInputs& in) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, kPoseDim, kPoseDim>> es(
      in.Lambda);
  // SelfAdjoint: ascending; u must follow the SAME order as eigenvalues.
  Eigen::Matrix<double, kPoseDim, kPoseDim> V = es.eigenvectors();  // cols
  Eigen::Matrix<double, kPoseDim, 1> lam = es.eigenvalues().cwiseMax(0.0);
  Eigen::Matrix<double, kPoseDim, 1> lam_g = lam.cwiseProduct(in.u.cwiseProduct(in.u));
  return V * lam_g.asDiagonal() * V.transpose();
}

// Per-direction information λ_k (of Λ_m) and the healthy-reference λ̄_k (of
// Λ_ref), in Λ_m's OWN eigenbasis — the direction coordinate in which the
// gate of prop:gate-optimal is defined (v_k fixed, both modalities report
// along it).
struct DirInfo {
  Eigen::Matrix<double, kPoseDim, 1> lam;        // λ_{m,k}, ascending order
  Eigen::Matrix<double, kPoseDim, 1> lam_ref;    // λ_{m̄,k} in the same basis
};
inline DirInfo directionInfos(const Eigen::Matrix<double, kPoseDim, kPoseDim>& Lambda_m,
                              const Eigen::Matrix<double, kPoseDim, kPoseDim>& Lambda_ref) {
  DirInfo d;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, kPoseDim, kPoseDim>> es(
      Lambda_m);
  d.lam = es.eigenvalues().cwiseMax(0.0);
  const auto& V = es.eigenvectors();
  d.lam_ref = (V.transpose() * Lambda_ref * V).diagonal().cwiseMax(0.0);
  return d;
}

// --- (b) Differential bias read (prop:bias-read B3) ---
// The differential read b̂^diff = r_L − r_V is unbiased for every λ and π
// (measured worst rel err 3.4e-4) with Var = 1/λ_L + 1/λ_V, π-free, and ≥2×
// the plug-in SNR everywhere. In-stream, per direction v_k of the fused
// information, the projected difference of the two modalities' scalarized
// NIS contributions is the implementable surrogate:
//   diff_k := nu_L,k − nu_V,k
// where nu_m,k is modality m's innovation energy projected on v_k. Under
// per-modality bias the E|diff| shifts by the bias difference; under
// COMMON-mode bias it stays at zero (prop:common-mode M2 — the read is
// blind there, which is the desired selectivity). EMA per direction.
struct DiffReadState {
  Eigen::Matrix<double, kPoseDim, 1> ema =
      Eigen::Matrix<double, kPoseDim, 1>::Zero();
  bool initialized = false;
};
// Update from instantaneous projected NIS difference (ascending eigorder of
// the fused probe's eigvecs — pass the same basis each frame).
inline void updateDiffRead(DiffReadState& s,
                           const Eigen::Matrix<double, kPoseDim, 1>& diff_inst,
                           double alpha) {
  if (!s.initialized) { s.ema = diff_inst; s.initialized = true; return; }
  s.ema = (1.0 - alpha) * s.ema + alpha * diff_inst;
}

// --- (c) External anchor (prop:common-mode M4, eq:anchor-att) ---
// The ONLY in-model defense against a steady common-mode bias: an unbiased
// pseudo-observation r_A = e + n_A with information λ_A, applied on ALL
// directions (not only the degenerate subspace — the common mode lives
// everywhere), attenuating it by (λ_L+λ_V)/(λ_L+λ_V+λ_A). Implemented as a
// PriorSource that assembles Λ_A = λ_A·I on the FULL 6-dim pose space with
// observation b_A = λ_A·z_A (the anchor residual, e.g. zero-motion or
// GNSS-style displacement).
class AnchorPrior : public PriorSource {
 public:
  AnchorPrior(double info, const Eigen::Matrix<double, kPoseDim, 1>& obs)
      : info_(info), obs_(obs) {}
  PriorObservation priorForDirection(int /*k*/,
                                     const DegeneracyResult& /*res*/) const override {
    PriorObservation po;
    po.valid = info_ > 0.0;
    po.info = info_;
    po.observation = obs_.mean();  // scalarized; see assembleAnchor below
    return po;
  }
 private:
  double info_;
  Eigen::Matrix<double, kPoseDim, 1> obs_;
};

// Assemble the anchor's full-rank information directly (bypasses the
// degenerate-subspace restriction of assemblePrior — the anchor must act on
// ALL directions to fight the common mode). Returns Λ_A = λ_A·I and
// b_A = λ_A·z_A.
inline void assembleAnchor(double lambda_A,
                           const Eigen::Matrix<double, kPoseDim, 1>& z_A,
                           Eigen::Matrix<double, kPoseDim, kPoseDim>& Lambda_A,
                           Eigen::Matrix<double, kPoseDim, 1>& b_A) {
  Lambda_A = lambda_A * Eigen::Matrix<double, kPoseDim, kPoseDim>::Identity();
  b_A = lambda_A * z_A;
}

}  // namespace rr_ieskf

}  // namespace dd_esikf

#endif  // DEGENERACY_H_
