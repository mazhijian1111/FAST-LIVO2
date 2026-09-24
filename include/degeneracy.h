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
// 0. Whitened-criterion configuration (section 2b below)
// ============================================================================
// Declared before DegeneracyConfig because the latter embeds it (whiten).
struct WhitenedConfig {
  // Dimensionless thresholds on w_k (units of prior information).
  // tau_w_abs = 0.25 is the pivot used for the covariance path: below it the
  // sensor adds less than a quarter of the information the prior already
  // carries. (1.0 is the "sensor adds nothing over the prior" reading — a
  // defensible alternative; 0.25 is chosen so the criterion has a usable
  // dynamic range on real data, where the ratio spans several decades.)
  double tau_w_abs = 0.25;
  // Relative reading, vs. w_1. DEFAULT 0.0 = the branch is OFF, and that is the
  // verified setting rather than a placeholder: w_1 is still a RATIO's reference,
  // and on real Outdoor04 it is 2.6e19 (median, live whitened run), so any
  // tau_w_rel > ~1e-17 hands the threshold to the relative branch and
  // "degenerate" fires on essentially every frame — 97.0% at 1e-9, 99.3% at
  // 1e-6, 100.0% (deg_rank ~4) at 1e-3. That is the §2.1 defect in different
  // clothes: whitening fixes the UNITS of the comparison, not the choice of
  // reference. The absolute branch is the one that means "the sensor is silent".
  // Set this > 0 only when the relative branch is itself the object of study.
  double tau_w_rel = 0.0;
  // Hard 0/1 mask instead of the continuous clip. Same semantics as
  // DegeneracyConfig::hard_mask.
  bool hard_mask = false;
  // Eigenvalue floor applied to P^- before the square root, so a
  // rank-deficient / numerically singular prior block cannot produce inf.
  double prior_ev_floor = 1e-12;
};

// Symmetric square root of an SPD matrix and of its inverse, with an eigenvalue
// floor so a singular prior block degrades gracefully instead of blowing up.
inline void symSqrtAndInverse(const Eigen::Matrix<double, kPoseDim, kPoseDim>& P,
                              double ev_floor,
                              Eigen::Matrix<double, kPoseDim, kPoseDim>& S,
                              Eigen::Matrix<double, kPoseDim, kPoseDim>& S_inv) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, kPoseDim, kPoseDim>> es(P);
  // cwiseMax guards a negative eigenvalue from round-off in a PSD input.
  const Eigen::Matrix<double, kPoseDim, 1> d =
      es.eigenvalues().cwiseMax(ev_floor);
  const Eigen::Matrix<double, kPoseDim, kPoseDim>& V = es.eigenvectors();
  S     = V * d.cwiseSqrt().asDiagonal() * V.transpose();
  S_inv = V * d.cwiseInverse().cwiseSqrt().asDiagonal() * V.transpose();
}

// ============================================================================
// 1. Configuration
// ============================================================================
struct DegeneracyConfig {
  // Dual degeneracy criterion (Definition 1 in theory.tex).
  // A direction v_k is degenerate iff  λ_k < tau_abs  OR  λ_k/λ_1 < tau_rel.
  //
  // IMPORTANT (fixed 2026-09-24): these are TWO READINGS OF ONE QUANTITY, not
  // two independent switches. The RHS of both branches is affine in λ_1, so
  //
  //     (λ_k < tau_abs) ∨ (λ_k/λ_1 < tau_rel)
  //       ⇔  λ_k < max(tau_abs, tau_rel·λ_1)  =:  τ_k .
  //
  // Previously τ_k was used for the degenerate/observable SPLIT (→ deg_indices,
  // Π_deg, and the covariance projection) while the MASK used tau_abs alone
  // (m_k = clip(λ_k/tau_abs, 0, 1)). On real data tau_abs never fires
  // (λ_L spans 1.2e2…9.2e11, so m_k ≡ 1 and M = I), while tau_rel fires on
  // 100% of frames — leaving the mean path a no-op and the covariance path
  // active, i.e. two inconsistent definitions of D inside a single probe
  // result. Both now use τ_k, so D = {k : m_k < 1} is a single set.
  double tau_abs = 1.0;   // absolute reading of τ_k (scene/sensor tuned)
  double tau_rel = 1e-3;  // relative reading of τ_k, vs. the largest eigenvalue

  // Mask shape: m_k = clip(λ_k / τ_k, 0, 1) with τ_k = max(tau_abs,
  // tau_rel·λ_1). A direction at the threshold gets m_k = 1 (unmasked); one at
  // λ_k = 0 gets m_k = 0. Continuous avoids hard-threshold chatter
  // (Proposition 5 / Assumption 1 need iterate-independence, and a continuous
  // mask also keeps Λ_eff PSD).
  //
  // Optional: hard-mask the weakly-observable directions for a stricter
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

  // ---- Whitened criterion (probeWhitened, section 2b below) ----------------
  // When true, `probe()` runs the criterion on the WHITENED information
  //   W = S Λ_f S,   S = (P^-)^{1/2}
  // and reports the WHITENED eigenvectors and projectors. The degenerate set D
  // is then {k : w_k < tau_w} instead of {k : λ_k < τ_k}, i.e. dimensionless
  // and invariant under error reparametrization (see the invariance note in
  // section 2b). Requires `prior_cov` to be filled by the caller (set it via
  // `setPriorCov`); if prior_cov_present is false (the default) the probe falls
  // back to the raw criterion, so nothing silently changes for callers that do
  // not supply a prior.
  bool whiten_enable = false;
  WhitenedConfig whiten;   // tau_w_abs / tau_w_rel / hard_mask / prior_ev_floor
  // Prior pose covariance P^- and its validity flag. `prior_cov_present` is the
  // gate: whiten_enable is a no-op without it.
  Eigen::Matrix<double, kPoseDim, kPoseDim> prior_cov =
      Eigen::Matrix<double, kPoseDim, kPoseDim>::Zero();
  bool prior_cov_present = false;
};

// ============================================================================
// 2. Degeneracy probe (Theorem 1 / Definition 1)
// ============================================================================
struct DegeneracyResult {
  // Spectral decomposition of Λ_L (6×6), eigenvalues in DESCENDING order.
  Eigen::Matrix<double, kPoseDim, 1> eigvals;   // λ_1 ≥ ... ≥ λ_6 ≥ 0
  Eigen::Matrix<double, kPoseDim, kPoseDim> eigvecs;  // columns = v_k
  Eigen::Array<bool, kPoseDim, 1> deg_mask;     // true => degenerate direction
  // The single effective threshold τ_k = max(tau_abs, tau_rel·λ_1) this probe
  // used. D = {k : λ_k < τ_k} = {k : m_k < 1}; logged for diagnosis.
  double tau_eff = 0.0;

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

  // ---- Whitened criterion (section 2b): set only when the probe was run in
  // whitened mode. `eigvals`/`eigvecs`/`M`/`Pi_obs`/`Pi_deg` above then refer to
  // the WHITENED quantities W = S Λ_f S and its eigenbasis — the eigenvectors
  // are ũ_k, and every direction is S^{-1}-related to the ξ-space direction
  // u_k = S ũ_k, which is the one the mask/prior should be assembled on.
  // `tau_whitened` records which criterion produced this result (logged).
  bool tau_whitened = false;
  Eigen::Matrix<double, kPoseDim, kPoseDim> whiten_S =
      Eigen::Matrix<double, kPoseDim, kPoseDim>::Identity();
  Eigen::Matrix<double, kPoseDim, kPoseDim> whiten_S_inv =
      Eigen::Matrix<double, kPoseDim, kPoseDim>::Identity();
  // ξ-space direction u_k = S ũ_k (unit PRIOR information: u_jᵀ(P^-)^{-1}u_k =
  // δ_jk). This is the basis the mask and the prior must be assembled in.
  Eigen::Matrix<double, kPoseDim, kPoseDim> xi_dirs =
      Eigen::Matrix<double, kPoseDim, kPoseDim>::Identity();
  // ξ-space projectors Π_ξ = S Π̃ S^{-1}. Idempotent, but NOT symmetric — they
  // are self-adjoint w.r.t. the prior metric (P^-)^{-1}. The covariance path in
  // voxel_map.cpp needs exactly these, plus the similarity
  //   P_ξ,pr = S P̃_ξ,pr S
  // that maps a whitened-coordinate covariance back to ξ. Both formulas are
  // derived in the wiring note next to the covariance projection.
  Eigen::Matrix<double, kPoseDim, kPoseDim> Pi_obs_xi =
      Eigen::Matrix<double, kPoseDim, kPoseDim>::Identity();
  Eigen::Matrix<double, kPoseDim, kPoseDim> Pi_deg_xi =
      Eigen::Matrix<double, kPoseDim, kPoseDim>::Zero();
};

// Compute the eigendecomposition of Λ_L and assemble the mask. Λ_L must be
// symmetric PSD (it is H^T R^-1 H, so this holds). Cost: one 6×6 SelfAdjoint
// eigen-decomposition per frame — microseconds.
//
// Two criteria are available, selected by `cfg.whiten_enable`:
//   false (default, raw):    λ_k of Λ_f,      threshold τ_k = max(tau_abs, tau_rel·λ_1)
//   true  (whitened, §2.2):  w_k of S Λ_f S,  threshold τ_w = max(tau_w_abs, tau_w_rel·w_1)
// Both produce the same fields. In the whitened case `eigvecs` holds the
// WHITENED eigenvectors ũ_k and the projectors are the whitened ones, so the
// SAME single D set still drives both the mean path (M) and the covariance path
// (Π_deg) — no second criterion anywhere. See section 2b for what each quantity
// means in the whitened case, and note that a consumer needing ξ-space
// projectors or a covariance-basis update must use `whiten_S`/`whiten_S_inv`
// (reported below and consumed by voxel_map.cpp).
inline DegeneracyResult probe(const Eigen::Matrix<double, kPoseDim, kPoseDim>& Lambda_L,
                              const DegeneracyConfig& cfg) {
  DegeneracyResult r;
  // Whiten once, up front: W = S Λ_f S. Everything below is the same code path
  // with W in place of Λ_f, because a symmetric congruence preserves every
  // structural property the raw probe relies on (symmetry, PSD-ness, and the
  // Rayleigh-quotient reading of the eigenvalues).
  const bool whiten = cfg.whiten_enable && cfg.prior_cov_present;
  Eigen::Matrix<double, kPoseDim, kPoseDim> Lambda_probe = Lambda_L;
  if (whiten) {
    symSqrtAndInverse(cfg.prior_cov, cfg.whiten.prior_ev_floor, r.whiten_S,
                      r.whiten_S_inv);
    Lambda_probe = r.whiten_S * Lambda_L * r.whiten_S;
    Lambda_probe = 0.5 * (Lambda_probe + Lambda_probe.transpose());
  }

  // Thresholds. Both branches collapse to a single max() for the same reason:
  // they are two readings of one quantity. In the whitened case both readings
  // are dimensionless (units of prior information), which is the whole point.
  const double t_abs = whiten ? cfg.whiten.tau_w_abs : cfg.tau_abs;
  const double t_rel = whiten ? cfg.whiten.tau_w_rel : cfg.tau_rel;
  const bool   hard  = whiten ? cfg.whiten.hard_mask : cfg.hard_mask;

  // SelfAdjointEigenSolver returns ASCENDING eigenvalues; we want DESCENDING
  // to match the theory's convention λ_1 ≥ ... ≥ λ_6.
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, kPoseDim, kPoseDim>> es(
      Lambda_probe);
  Eigen::Matrix<double, kPoseDim, 1> evals = es.eigenvalues();        // ascending
  Eigen::Matrix<double, kPoseDim, kPoseDim> evecs = es.eigenvectors();  // cols

  // Reverse to descending order.
  for (int i = 0; i < kPoseDim; ++i) {
    r.eigvals(i) = evals(kPoseDim - 1 - i);
    r.eigvecs.col(i) = evecs.col(kPoseDim - 1 - i);
  }

  const double lam1 = r.eigvals(0);
  const double tau_k = std::max(t_abs, t_rel * std::max(lam1, 1e-12));
  r.tau_eff = tau_k;
  r.tau_whitened = whiten;
  r.deg_mask.resize(kPoseDim);
  r.m_coefs.setZero();
  r.M.setZero();
  for (int k = 0; k < kPoseDim; ++k) {
    const double lk = r.eigvals(k);
    // Continuous mask: m_k = clip(λ_k / τ_k, 0, 1). Reference-scaled:
    // λ_k = τ_k ⇒ m_k = 1 (barely observable, kept), λ_k = 0 ⇒ m_k = 0.
    const double mk_raw = std::min(1.0, std::max(0.0, lk / std::max(tau_k, 1e-12)));
    const bool deg = hard ? (mk_raw < 1.0) : (lk < tau_k);
    r.deg_mask(k) = deg;
    r.m_coefs(k) = hard ? (deg ? 0.0 : 1.0) : mk_raw;
    r.M += r.m_coefs(k) * (r.eigvecs.col(k) * r.eigvecs.col(k).transpose());
    if (deg) r.deg_indices.push_back(k);
    else     r.obs_indices.push_back(k);
  }
  if (!r.deg_indices.empty()) {
    r.idx_weak   = r.deg_indices.front();   // largest λ_k among degenerate
    r.idx_strict = r.deg_indices.back();    // smallest λ_k
  }
  // Layer 3 projectors: Π_obs over observable, Π_deg = I − Π_obs. With the
  // single-threshold fix, Π_deg's support is exactly {k : m_k < 1}, so the
  // mean path (M) and the covariance path (Π_deg) act on the SAME subspace D.
  r.Pi_obs.setZero();
  for (int k : r.obs_indices)
    r.Pi_obs += r.eigvecs.col(k) * r.eigvecs.col(k).transpose();
  r.Pi_deg = Eigen::Matrix<double, kPoseDim, kPoseDim>::Identity() - r.Pi_obs;

  // ---- Whitened mode: everything above is in ξ̃ = S^{-1}ξ coordinates. Map the
  // directions and the projectors back to ξ once, here, so no consumer has to
  // know which criterion ran. ----
  if (whiten) {
    for (int k = 0; k < kPoseDim; ++k)
      r.xi_dirs.col(k) = r.whiten_S * r.eigvecs.col(k);
    r.Pi_obs_xi = r.whiten_S * r.Pi_obs * r.whiten_S_inv;
    r.Pi_deg_xi = r.whiten_S * r.Pi_deg * r.whiten_S_inv;
  }
  return r;
}

// ============================================================================
// 2b. Whitened (prior-normalized) degeneracy criterion
// (WhitenedConfig and symSqrtAndInverse are declared in section 0, above, so
//  that DegeneracyConfig can embed the former and probe() can call the latter.)
// ============================================================================
// WHY. The raw criterion λ_k < τ_k compares a LiDAR information eigenvalue
// against a threshold built from λ_1. Both carry MIXED PHYSICAL UNITS: the
// rotational block of Λ_L is in rad^-2, the translational block in m^-2, and
// λ_1 — the reference of the relative branch — is on real data a ROTATIONAL
// eigenvalue (its eigenvector's position block has norm ≈ 0.056, see the
// Outdoor04 probe log). So the translational degeneracy decision currently
// depends on rotational observability, which is dimensionally meaningless: a
// scene that merely rotates better re-classifies translation directions.
//
// WHAT. Whiten by the prior pose covariance. Let S = (P^-)^{1/2} be the
// symmetric square root of the 6×6 prior pose covariance. In the error
// coordinates ξ̃ := S^{-1} ξ the prior covariance is the IDENTITY, since
//     cov(S^{-1}ξ) = S^{-1} P^- S^{-1} = I   (as (P^-)^{-1} = S^{-1}S^{-1}),
// and the information matrix transforms as Λ̃_L = S Λ_L S. So
//     W := S Λ_L S ,
//     w_k := eig_k(W) = sensor information along direction k, in units of the
//                       prior information along that same direction,
// a dimensionless number. w_k = 1 means the sensor contributes exactly as much
// information as the prior already carries (a factor-2 variance reduction);
// w_k ≪ 1 means the direction is truly unobservable — the sensor cannot improve
// on the prior — which is the estimation-theoretic meaning of degeneracy. This
// is sensor-, scene- and unit-independent: only the RATIO of sensor information
// to prior information matters.
//
// INVARIANCE (the property the raw criterion lacks, and the one worth proving).
// Under any invertible error reparametrization ξ → ξ' = T ξ, with
//     Λ' = T^{-T} Λ T^{-1},   P' = T P T^T,
// we have W' = (P')^{1/2} Λ' (P')^{1/2} orthogonally similar to W, i.e.
//     spec(W') = spec(W)   exactly,
// because W ~ ΛP (similarity) and Λ'P' ~ ΛP. Consequently the ORDERED
// spectrum, the flag pattern {k : w_k < τ_w}, the degenerate rank and the
// whitened projectors (up to the same orthogonal congruence) are all invariant.
// The raw criterion has no such property: it is tied to the eigenbasis of Λ_L,
// whose eigenvalues are not invariant under ξ → Tξ at all.
//
// LIMITATION (honest). With a block-diagonal Λ_L and a block-diagonal prior,
// W is block-diagonal, so rotational and translational decisions decouple
// EXACTLY — which is what fixes the defect above. In general Λ_L has a non-zero
// θ–p cross block and the decoupling is only approximate. The invariance above
// holds regardless.
//
// SECOND LIMITATION, worth stating loudly because it changes what the mask
// means. w_k ≪ 1 is "the sensor is silent". It is NOT the same as "the sensor
// is unreliable". On a median Outdoor04 frame the translational information is
// ~10^9 × the prior information, so NO direction is prior-degenerate: whitening
// says the LiDAR is never silent there. The failure mode in a corridor is that
// the weak direction's residual is BIASED (the plane fits the wrong surface),
// not that its information is small. A criterion built on w_k alone therefore
// cannot fire on Outdoor04's median — by construction, and correctly. Firing on
// bias is the job of the residual/bias channel (b-side gate, thm:residual,
// cor:tilt-resolution), and the two channels are complementary: w_k decides
// "is there information", the bias read decides "is the information credible".

struct WhitenedResult {
  // W = S Λ_L S, and its descending spectrum/eigenbasis (ũ_k live in the
  // whitened error coordinates ξ̃).
  Eigen::Matrix<double, kPoseDim, kPoseDim> W;
  Eigen::Matrix<double, kPoseDim, 1> w_vals;   // w_1 ≥ … ≥ w_6 ≥ 0
  Eigen::Matrix<double, kPoseDim, kPoseDim> w_vecs;  // columns = ũ_k

  // The same directions mapped back to ξ-space: u_k = S ũ_k. These are the
  // ξ-perturbations the whitened problem is about, and they carry UNIT PRIOR
  // INFORMATION by construction: u_j^T (P^-)^{-1} u_k = δ_jk, so w_k =
  // u_k^T Λ_L u_k is read as "sensor information per unit prior information".
  // Under ξ → Tξ they transport geometrically as u'_k = T u_k. Not orthonormal
  // in the Euclidean metric of ξ (orthonormal in the prior metric) — use them
  // for reporting/tests; the update should consume the projectors below.
  Eigen::Matrix<double, kPoseDim, kPoseDim> u_dirs;

  // S = (P^-)^{1/2} and S^{-1}, cached for the caller.
  Eigen::Matrix<double, kPoseDim, kPoseDim> S, S_inv;

  // Effective dimensionless threshold tau_w = max(tau_w_abs, tau_w_rel·w_1).
  double tau_eff = 0.0;

  Eigen::Array<bool, kPoseDim, 1> deg_mask;
  Eigen::Matrix<double, kPoseDim, 1> m_coefs;   // m_k = clip(w_k/tau_w, 0, 1)
  std::vector<int> deg_indices, obs_indices;

  // Projectors in the WHITENED coordinates (symmetric, orthonormal basis ũ).
  Eigen::Matrix<double, kPoseDim, kPoseDim> Pi_obs_tilde, Pi_deg_tilde;
  // The same projectors acting on ξ: S Π̃ S^{-1}. These are projectors
  // ((SΠ̃S^{-1})² = SΠ̃S^{-1}) but NOT symmetric — they are self-adjoint with
  // respect to the prior metric (P^-)^{-1}. The DD machinery in voxel_map.cpp
  // currently assumes symmetric Π; consuming these requires either working in
  // ξ̃ throughout or carrying the metric explicitly. Flagged, not papered over.
  Eigen::Matrix<double, kPoseDim, kPoseDim> Pi_obs_xi, Pi_deg_xi;

  bool has_degeneracy() const { return !deg_indices.empty(); }
  int deg_rank() const { return deg_indices.size(); }
};

// Whitened probe. `Lambda_L` = H^T R^-1 H (or Λ_L + Λ_V for the fused probe —
// the whitening is a congruence on the fused information either way, and the
// P1 inclusion argument is unchanged); `PriorCov` = P^-, the 6×6 prior pose
// covariance from propagation (voxel_map.cpp: state_propagat.cov.block<6,6>).
inline WhitenedResult probeWhitened(
    const Eigen::Matrix<double, kPoseDim, kPoseDim>& Lambda_L,
    const Eigen::Matrix<double, kPoseDim, kPoseDim>& PriorCov,
    const WhitenedConfig& cfg) {
  WhitenedResult r;
  symSqrtAndInverse(PriorCov, cfg.prior_ev_floor, r.S, r.S_inv);

  r.W = r.S * Lambda_L * r.S;
  // Re-symmetrize: the congruence is symmetric in exact arithmetic; this only
  // removes floating-point asymmetry so the eigensolver sees a symmetric input.
  r.W = 0.5 * (r.W + r.W.transpose());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, kPoseDim, kPoseDim>> es(r.W);
  const Eigen::Matrix<double, kPoseDim, 1> evals = es.eigenvalues();       // ascending
  const Eigen::Matrix<double, kPoseDim, kPoseDim> evecs = es.eigenvectors();
  for (int i = 0; i < kPoseDim; ++i) {   // reverse to descending
    r.w_vals(i)   = evals(kPoseDim - 1 - i);
    r.w_vecs.col(i) = evecs.col(kPoseDim - 1 - i);
  }

  // Same single-threshold collapse as the raw probe, now in dimensionless
  // units: (w_k < tau_w_abs) ∨ (w_k/w_1 < tau_w_rel) ⇔ w_k < max(tau_w_abs,
  // tau_w_rel·w_1).
  const double w1 = r.w_vals(0);
  const double tau_w = std::max(cfg.tau_w_abs, cfg.tau_w_rel * std::max(w1, 1e-12));
  r.tau_eff = tau_w;

  r.deg_mask.resize(kPoseDim);
  r.m_coefs.setZero();
  for (int k = 0; k < kPoseDim; ++k) {
    const double wk = r.w_vals(k);
    const double mk_raw = std::min(1.0, std::max(0.0, wk / std::max(tau_w, 1e-12)));
    const bool deg = cfg.hard_mask ? (mk_raw < 1.0) : (wk < tau_w);
    r.deg_mask(k) = deg;
    r.m_coefs(k)  = cfg.hard_mask ? (deg ? 0.0 : 1.0) : mk_raw;
    if (deg) r.deg_indices.push_back(k);
    else     r.obs_indices.push_back(k);
    r.u_dirs.col(k) = r.S * r.w_vecs.col(k);
  }

  r.Pi_obs_tilde.setZero();
  for (int k : r.obs_indices)
    r.Pi_obs_tilde += r.w_vecs.col(k) * r.w_vecs.col(k).transpose();
  r.Pi_deg_tilde =
      Eigen::Matrix<double, kPoseDim, kPoseDim>::Identity() - r.Pi_obs_tilde;
  r.Pi_obs_xi = r.S * r.Pi_obs_tilde * r.S_inv;
  r.Pi_deg_xi = r.S * r.Pi_deg_tilde * r.S_inv;
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
  // NOTE: do NOT early-return when `src` is null. The default_prior_info
  // floor below must be applied consistently in BOTH the mean path (here:
  // Λ_eff / Htz_eff) and the covariance path (voxel_map.cpp Schur floor) —
  // gating the floor on a wired PriorSource made the covariance use prior
  // information the mean never saw (mean/covariance inconsistency).
  if (!res.has_degeneracy()) return;
  bool src_gave_any = false;
  if (src) {
    for (int k : res.deg_indices) {
      const PriorObservation po = src->priorForDirection(k, res);
      if (!po.valid || po.info <= 0.0) continue;
      const auto v_k = res.eigvecs.col(k);
      Lambda_pr += po.info * (v_k * v_k.transpose());
      b_pr += po.info * po.observation * v_k;
      src_gave_any = true;
    }
  }
  // Fallback default floor on degenerate directions when no source is wired
  // OR the source gave nothing on any degenerate direction.
  if (cfg.default_prior_info > 0.0 && !src_gave_any) {
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

  // ---- Whitened mode: consumed by the caller when probe_res.tau_whitened ----
  // In whitened mode `probe_res.M` is built in the ξ̃ directions ũ_k = S^{-1}u_k,
  // so it is NOT the operator acting on ξ. The ξ-space mask operator is the
  // SIMILARITY  M_ξ = S^{-1} M̃ S  (not the congruence S M̃ S — that is a
  // different operator with a different spectrum, and it does not even have the
  // right quadratic form). Numerically with a physically-scaled prior
  // (σ_θ = 5e-4 rad, σ_p = 1e-2 m) and the 3 translational directions
  // degenerate: eig(S M̃ S) = {0,0,0, 1e-4,1e-4,1e-4}, ‖S M̃ S‖ = 1.7e-4,
  // while eig(S^{-1}M̃S) = {0,0,0, 1,1,1}, ‖S^{-1}M̃S‖ = 1.7, and only the
  // latter reproduces vᵀΛ_eff v = ṽᵀW_eff ṽ. Λ_eff below is therefore built
  // through the whitened space, not by masking Λ_L in place.
  Eigen::Matrix<double, kPoseDim, kPoseDim> M_xi =
      Eigen::Matrix<double, kPoseDim, kPoseDim>::Identity();
  // WHITENED-coordinate (ξ̃) projectors — the well-conditioned pair. The
  // covariance path uses these and converts the ξ-space prior covariance into
  // ξ̃ with P̃ = S^{-1} P S^{-1} (inverse congruence), applies the standard
  // theorem there, and converts back with P = S P̃ S. Both the projection and
  // the congruence keep PSD, which is the point: operating on the ill-scaled
  // ξ-space Π_ξ would not.
  Eigen::Matrix<double, kPoseDim, kPoseDim> Pi_obs_tilde =
      Eigen::Matrix<double, kPoseDim, kPoseDim>::Identity();
  Eigen::Matrix<double, kPoseDim, kPoseDim> Pi_deg_tilde =
      Eigen::Matrix<double, kPoseDim, kPoseDim>::Zero();
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
  //
  // Whitened mode: `probe_res.M` lives in ξ̃, so route through the whitened
  // space instead of masking Λ_L in place:
  //     W = S Λ_L S,  W_eff = M̃ W M̃,  Λ_eff = S^{-1} W_eff S^{-1}
  // which is exactly M_ξ Λ_L M_ξᵀ with M_ξ = S^{-1}M̃S. Λ_pr is assembled in ξ
  // (assemblePrior uses probe_res.xi_dirs), so it is added AFTER the map back.
  if (out.probe_res.tau_whitened) {
    const auto& S    = out.probe_res.whiten_S;
    const auto& Sinv = out.probe_res.whiten_S_inv;
    const auto& Mt   = out.probe_res.M;
    out.M_xi = Sinv * Mt * S;
    const Eigen::Matrix<double, kPoseDim, kPoseDim> W =
        0.5 * ((S * in.Lambda_L * S) + (S * in.Lambda_L * S).transpose());
    const Eigen::Matrix<double, kPoseDim, kPoseDim> W_eff = Mt * W * Mt;
    out.Lambda_eff = 0.5 * ((Sinv * W_eff * Sinv) +
                            (Sinv * W_eff * Sinv).transpose()) + Lambda_pr;
    // Residual information vector: the same map on the b side. b transforms
    // like an information vector under ξ = S ξ̃, i.e. b_ξ = S^{-1} b_ξ̃, so the
    // ξ-space masked residual is M_ξ (S^{-1} M̃ S) applied to Htz… which is
    // M_ξ Htz only if Htz too is read in ξ, which it is. Keep the symmetric
    // form: b_eff = M_ξ Htz + b_pr.
    out.Htz_eff = out.M_xi * in.Htz + b_pr;
    out.Pi_obs_tilde = out.probe_res.Pi_obs;
    out.Pi_deg_tilde = out.probe_res.Pi_deg;
  } else {
    out.Lambda_eff = out.probe_res.M * in.Lambda_L * out.probe_res.M + Lambda_pr;
    out.Htz_eff    = out.probe_res.M * in.Htz + b_pr;
  }
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
