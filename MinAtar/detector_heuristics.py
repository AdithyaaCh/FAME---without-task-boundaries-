"""Pre-run parameter heuristics for the TSDM family (SWOKS / implicit / hybrid).

Two entry points are exposed:

    suggest_params(switch_steps, mini=False)
        Lightweight, backwards-compatible API used by FAME.py --auto_params.
        Only needs the expected task length; returns a DetectorParams object.

    suggest_params_full(**ctx)
        Richer API used by experiment.py: also consumes the TOTAL budget
        (num_tasks * switch_steps), the action space size and the latent-
        feature dimension.  Produces parameters that are scaled with the
        full-run multiple-testing burden (Bonferroni-style alpha), and with
        the representational capacity requirements of the implicit TSN.

Design principles (shared by both APIs)
---------------------------------------

1. **Windows scale sub-linearly with the task.**
   A task of 500k steps needs a larger detection window than a 5k-step task,
   but the ratio shouldn't be constant: very short tasks have so little data
   that we must still preserve >= ~100 post-shift samples for the statistic
   to have any power.  `L_D = clamp(switch/40, 100, 1500)`.

2. **Detection interval = O(L_D / 5).**
   Running the test too often wastes compute and tightens the multiple-
   testing correction; too rarely and detection latency grows.  `L_D / 5`
   gives ~5 overlapping tests per window.

3. **Stable phase = ~ task_length / 15.**
   After firing, wait long enough for the reference window to fill with
   post-shift data, but not so long that short back-to-back shifts are lost.

4. **Warmup >= 3 * L_D.**
   Features / TSN must settle before their errors carry signal.

5. **Alpha is family-wise-corrected.**
   If the detector performs ~N tests over a full run, pick alpha such that
   the expected false-positive count is bounded.  For the paper's 3.5M-step
   MinAtar protocol with detection_interval=240, N ~ 14,500 -- a target of
   expected FP = 1 implies alpha ~ 7e-5.  We cap at 1e-3 to stay sensitive.

6. **Hybrid combined threshold = -log(alpha_imp) - log(alpha_stat).**
   Setting tau_combined equal to the sum of single-detector log-thresholds
   makes the combined-nats gate statistically equivalent to "both detectors
   at their own alpha" under independence, but realised as a SUM of log-
   p-values so one strong signal can compensate for a weaker co-signal.

7. **Implicit update cadence scales with replay churn.**
   Target ~1 TSN gradient step per batch-size's worth of fresh transitions.

These rules are *suggestions*, not hard requirements -- they are what
FAME.py --auto_params dials in if the user hasn't overridden detector flags.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional


# ----------------------------------------------------------------------
# Container
# ----------------------------------------------------------------------
@dataclass
class DetectorParams:
    # Shared across SWOKS / implicit / hybrid
    L_D: int
    L_W: int
    detection_interval: int
    warmup: int
    stable_phase: int
    # SWOKS
    alpha: float
    beta: float
    # Implicit
    imp_alpha: float
    imp_lr: float
    imp_buffer: int
    imp_update_every: int
    imp_batch_size: int = 64
    imp_hidden: int = 64
    # Hybrid
    hyb_tau_imp_loose: float = 1e-2
    hyb_tau_imp_strict: float = 1e-6
    hyb_tau_stat_strict: float = 1e-3
    hyb_tau_stat_failsafe: float = 1e-6
    hyb_tau_combined: float = 15.0
    hyb_horizon: int = 480
    hyb_persistence: int = 2
    # Book-keeping (not directly used by detectors, but useful for logging)
    expected_num_tests: int = 0
    expected_fp_count: float = 0.0

    def as_dict(self) -> Dict:
        return asdict(self)


# ----------------------------------------------------------------------
# Public helpers
# ----------------------------------------------------------------------
def fp_corrected_alpha(num_tests: int,
                       target_expected_fp: float = 1.0,
                       cap: float = 1e-3,
                       floor: float = 1e-6) -> float:
    """Return an alpha whose expected-FP over `num_tests` tests is target.

    Equivalent to Bonferroni: alpha = target / num_tests, then clipped into
    [floor, cap].  `cap` keeps the detector sensitive on short runs, `floor`
    avoids vanishing to zero on astronomically long ones.
    """
    if num_tests <= 0:
        return cap
    alpha = target_expected_fp / float(num_tests)
    return max(floor, min(cap, alpha))


def combined_nats_from(alpha_imp: float, alpha_stat: float,
                       safety: float = 0.0) -> float:
    """-log(alpha_imp) - log(alpha_stat) + safety.

    Setting tau_combined to this sum makes the combined gate equivalent to
    "both detectors at their own alpha" under independence, realised as the
    log-evidence sum.  `safety` is an extra margin (nats) that tightens the
    gate beyond the naive product.
    """
    return (-math.log(max(alpha_imp, 1e-300))
            - math.log(max(alpha_stat, 1e-300))
            + safety)


# ----------------------------------------------------------------------
# Legacy light API (kept so existing --auto_params path still works)
# ----------------------------------------------------------------------
def suggest_params(switch_steps: int, *, mini: bool = False) -> DetectorParams:
    """Legacy API: task-horizon-only heuristic."""
    total = switch_steps * 7  # assume 7-task MinAtar unless caller knows better
    return suggest_params_full(
        switch_steps=switch_steps,
        total_steps=total,
        num_actions=6,
        latent_dim=256,
        mini=mini,
    )


# ----------------------------------------------------------------------
# Richer API
# ----------------------------------------------------------------------
def suggest_params_full(
    *,
    switch_steps: int,
    total_steps: Optional[int] = None,
    num_actions: int = 6,
    latent_dim: int = 256,
    target_expected_fp: float = 1.0,
    mini: bool = False,
) -> DetectorParams:
    """Return heuristic parameters scaled to run size + network size.

    Parameters
    ----------
    switch_steps : int
        Expected number of steps between task switches (args.switch).
    total_steps : int, optional
        Expected total number of training steps across the whole sequence.
        Drives the Bonferroni-style alpha correction.  Defaults to
        7 * switch_steps (paper's K=7 protocol).
    num_actions : int
        Action space cardinality (only used to right-size the implicit TSN
        batch size: larger action sets need more per-step samples).
    latent_dim : int
        Dimensionality of the policy's penultimate feature.  Larger latents
        benefit from a slightly larger batch size.
    target_expected_fp : float
        Target expected number of false positives over the whole run.
        Default 1.0 (at most one spurious detection per training run).
    mini : bool
        If True, loosen alpha to 5e-2 and beta to 1.2 for short runs.
    """
    if total_steps is None:
        total_steps = max(switch_steps * 7, switch_steps)

    # -------- Windows ------------------------------------------------
    L_D = max(100, min(int(switch_steps) // 40, 1500))
    # L_W chosen so the reference window reaches ~L_W*L_D into the past --
    # this must be smaller than switch_steps so the reference can actually
    # pre-date the next shift.
    L_W_cap = max(5, min(30, int(switch_steps) // (2 * L_D)))
    L_W = max(5, L_W_cap)

    detection_interval = max(20, L_D // 5)
    warmup = max(500, L_D * 3)
    stable_phase = max(1000, int(switch_steps) // 15)

    # -------- Family-wise alpha -------------------------------------
    num_tests = max(1, int(total_steps) // detection_interval)
    if mini:
        alpha = 5e-2
        imp_alpha = 5e-2
    else:
        alpha = fp_corrected_alpha(num_tests, target_expected_fp,
                                   cap=1e-3, floor=1e-5)
        imp_alpha = fp_corrected_alpha(num_tests, target_expected_fp,
                                       cap=1e-3, floor=1e-5)

    beta = 1.2 if mini else 2.0

    # -------- Implicit detector sizing ------------------------------
    imp_lr = 1e-4
    # Replay must comfortably hold at least two full windows so that we can
    # train on fresh data without immediately forgetting the last regime.
    imp_buffer = max(4 * L_D, 2000)
    # Batch size grows gently with latent_dim (MSE noise ~ sqrt(latent_dim)).
    imp_batch_size = int(max(64, min(256, 64 + 4 * int(math.sqrt(latent_dim)))))
    # Update cadence: ~1 gradient step per batch_size fresh transitions.
    imp_update_every = max(4, min(32, max(4, imp_batch_size // 4)))

    # Hidden size scales linearly with the joint (phi, a) input breadth, but
    # we keep it in a ballpark small enough that the TSN stays dirt cheap.
    imp_hidden = int(max(64, min(128, 32 + latent_dim // 4 + num_actions)))

    # -------- Hybrid thresholds -------------------------------------
    # Loose entry: a single-tailed 5% quantile in mini runs, 1% in full runs.
    hyb_tau_imp_loose = 5e-2 if mini else 1e-2
    # Strict bypasses = standalone single-detector alphas.
    hyb_tau_imp_strict = 1e-4 if mini else 1e-6
    hyb_tau_stat_strict = alpha
    # Pure-stat failsafe must be stricter than stat_strict to avoid double
    # firing paths; use a factor-10 tighter gate.
    hyb_tau_stat_failsafe = max(1e-9, alpha * 1e-1)
    # Combined = sum of single-detector -log alphas + small safety (2 nats).
    hyb_tau_combined = combined_nats_from(hyb_tau_imp_loose,
                                          hyb_tau_stat_strict,
                                          safety=2.0)
    hyb_horizon = max(2 * detection_interval, stable_phase // 80)
    hyb_persistence = 2

    return DetectorParams(
        L_D=L_D, L_W=L_W,
        detection_interval=detection_interval,
        warmup=warmup,
        stable_phase=stable_phase,
        alpha=alpha, beta=beta,
        imp_alpha=imp_alpha,
        imp_lr=imp_lr,
        imp_buffer=imp_buffer,
        imp_update_every=imp_update_every,
        imp_batch_size=imp_batch_size,
        imp_hidden=imp_hidden,
        hyb_tau_imp_loose=hyb_tau_imp_loose,
        hyb_tau_imp_strict=hyb_tau_imp_strict,
        hyb_tau_stat_strict=hyb_tau_stat_strict,
        hyb_tau_stat_failsafe=hyb_tau_stat_failsafe,
        hyb_tau_combined=hyb_tau_combined,
        hyb_horizon=hyb_horizon,
        hyb_persistence=hyb_persistence,
        expected_num_tests=num_tests,
        expected_fp_count=alpha * num_tests,
    )


def format_suggestion(params: DetectorParams, switch_steps: int) -> str:
    lines = [f"Heuristic params for switch={switch_steps}:"]
    for k, v in params.as_dict().items():
        if isinstance(v, float):
            lines.append(f"  {k:<24s} = {v:.3g}")
        else:
            lines.append(f"  {k:<24s} = {v}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI sanity
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    for s in (4000, 8000, 32000, 500000, 3500000):
        p = suggest_params_full(switch_steps=s,
                                total_steps=s * 7,
                                mini=(s < 50000))
        sys.stdout.write(format_suggestion(p, s) + "\n\n")
