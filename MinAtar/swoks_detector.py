import math
from collections import deque

import numpy as np
from scipy import stats

try:
    import ot  # Python Optimal Transport
    _HAS_OT = True
except Exception:
    _HAS_OT = False


def _sliced_wasserstein(X, Y, n_projections=50, rng=None):
    """Sliced Wasserstein Distance between two equal-size samples.

    Falls back to a NumPy implementation when the `ot` package is not
    available (this keeps the detector runnable in a minimal CPU env).
    """
    if _HAS_OT:
        seed = None if rng is None else int(rng.integers(0, 2**31 - 1))
        return float(
            ot.sliced_wasserstein_distance(
                X, Y, n_projections=n_projections, seed=seed
            )
        )

    # Manual SWD: project onto random unit directions, sort, average L2.
    rng = rng if rng is not None else np.random.default_rng(0)
    d = X.shape[1]
    projs = rng.standard_normal((n_projections, d))
    projs /= (np.linalg.norm(projs, axis=1, keepdims=True) + 1e-12)
    Xp = np.sort(X @ projs.T, axis=0)
    Yp = np.sort(Y @ projs.T, axis=0)
    return float(np.mean(np.linalg.norm(Xp - Yp, axis=0) / np.sqrt(X.shape[0])))


class SwoksDetector:
    """Online, statistics-only boundary detector.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the penultimate feature phi used in detection.
    L_D : int
        Size of each window (number of transitions).  Proposal default = 1200.
    L_W : int
        Length of the SWD history on each side of the KS test.
    alpha : float
        Significance level for the KS test.
    beta : float
        Multiplicative adjustment applied to the old SWD history (>=1).
    stable_phase : int
        Suppress new detections for this many steps after every shift.
    detection_interval : int
        Compute SWD once every `detection_interval` steps.  Avoids cost
        of per-step testing and matches SWOKS' 240-step cadence.
    warmup : int
        Number of initial steps without any detection (lets phi stabilise).
    n_projections : int
        Number of slices for SWD.
    max_wait : int
        If > 0, allow a forced fallback probe after this many steps without
        detection (guards against false negatives from an over-strong beta).
    seed : int
        RNG seed.
    """

    def __init__(
        self,
        latent_dim,
        L_D=1200,
        L_W=30,
        alpha=1e-3,
        beta=2.0,
        stable_phase=36000,
        detection_interval=240,
        warmup=5000,
        n_projections=50,
        max_wait=0,
        seed=0,
    ):
        self.latent_dim = latent_dim
        self.L_D = int(L_D)
        self.L_W = int(L_W)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.stable_phase = int(stable_phase)
        self.detection_interval = int(detection_interval)
        self.warmup = int(warmup)
        self.n_projections = int(n_projections)
        self.max_wait = int(max_wait)

        self.rng = np.random.default_rng(seed)

        # Per-step detection features.  Capacity follows the SWOKS reference
        # implementation: L_D * (L_W + 1) items, so the reference window can
        # lag the current window by up to L_W * L_D steps.  That gives the
        # detector a ~L_W*L_D wide window in which a shift can be seen;
        # after that the reference itself fills with post-shift data and
        # the stream looks stationary again.
        self._buffer = deque(maxlen=self.L_D * (self.L_W + 1))
        # History of SWD scalar values.  We test the oldest L_W vs. newest L_W.
        self._swd_hist = deque(maxlen=2 * self.L_W)

        self.ts = 0
        self.last_shift_step = 0
        self.detections = []          # list of absolute time steps
        self.last_pval = 1.0
        self.last_swd = 0.0
        # Fire-time snapshot preserved across `_on_detected()` reset.
        self.last_fire_info = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def step(self, phi, action, reward):
        """Feed one transition into the detector.

        Parameters
        ----------
        phi : np.ndarray (1-D)
            Penultimate feature vector from the fast learner on s_t.
        action : int or array-like
            Action taken at time t.
        reward : float
            Reward received at time t.

        Returns
        -------
        bool
            True iff a task shift was *newly* detected at this step.
        """
        self.ts += 1

        feat = self._make_feature(phi, action, reward)
        self._buffer.append(feat)

        # Only compute SWD on a schedule.
        if self.ts < self.warmup:
            return False
        if self.ts - self.last_shift_step < self.stable_phase:
            return False
        if self.ts % self.detection_interval != 0:
            return False
        # Need at least 2*L_D items before ref and cur can be disjoint.
        if len(self._buffer) < 2 * self.L_D:
            return False

        buf = np.asarray(self._buffer, dtype=np.float32)
        ref = buf[: self.L_D]          # oldest in buffer (lags by ~len-L_D)
        cur = buf[-self.L_D :]          # newest
        swd = _sliced_wasserstein(
            ref, cur, n_projections=self.n_projections, rng=self.rng
        )
        self.last_swd = swd
        self._swd_hist.append(swd)

        # Need two full L_W halves to run the KS test.
        if len(self._swd_hist) < 2 * self.L_W:
            return False

        hist = np.asarray(self._swd_hist, dtype=np.float64)
        old = hist[: self.L_W] * self.beta
        new = hist[self.L_W :]
        # One-sided KS test following SWOKS's argument convention:
        # ks_2samp(x1=old*beta, x2=new, alternative="greater") has
        #   H1 : F_{old*beta}(x) > F_{new}(x)  <=>  new stochastically > old*beta.
        # We fire when that alternative is accepted (small p-value).
        _, pval = stats.ks_2samp(old, new, alternative="greater")
        self.last_pval = pval

        fired = pval < self.alpha
        if not fired and self.max_wait > 0:
            # Optional FN fallback: forced probe after a long quiet period.
            quiet = self.ts - self.last_shift_step
            # Compare against the UNADJUSTED old mean for the fallback.
            old_raw = hist[: self.L_W]
            if quiet > self.max_wait and new.mean() > 1.5 * old_raw.mean():
                fired = True

        if fired:
            self.last_fire_info = {
                "step": self.ts,
                "pval": self.last_pval,
                "swd": self.last_swd,
                "new_mean": float(new.mean()),
                "old_mean": float(hist[: self.L_W].mean()),
            }
            self._on_detected()
        return fired

    def reset_after_boundary(self):
        """Externally callable reset, e.g. when the oracle fires for debug."""
        self._on_detected()

    def stats(self):
        return {
            "ts": self.ts,
            "last_shift_step": self.last_shift_step,
            "num_detections": len(self.detections),
            "last_pval": self.last_pval,
            "last_swd": self.last_swd,
            "buffer_size": len(self._buffer),
            "swd_hist_size": len(self._swd_hist),
        }

    # ------------------------------------------------------------------
    # Serialisation -- used by FAME.py mid-training checkpointing
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """Return a picklable snapshot of all mutable detector state."""
        return {
            "buffer": list(self._buffer),          # list[np.ndarray]
            "swd_hist": list(self._swd_hist),       # list[float]
            "ts": self.ts,
            "last_shift_step": self.last_shift_step,
            "detections": list(self.detections),
            "last_pval": self.last_pval,
            "last_swd": self.last_swd,
            "last_fire_info": dict(self.last_fire_info),
            "rng": self.rng.__getstate__(),
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore mutable state from a previously saved state_dict."""
        self._buffer = deque(sd["buffer"], maxlen=self._buffer.maxlen)
        self._swd_hist = deque(sd["swd_hist"], maxlen=self._swd_hist.maxlen)
        self.ts = sd["ts"]
        self.last_shift_step = sd["last_shift_step"]
        self.detections = list(sd["detections"])
        self.last_pval = sd["last_pval"]
        self.last_swd = sd["last_swd"]
        self.last_fire_info = dict(sd["last_fire_info"])
        self.rng.__setstate__(sd["rng"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_feature(self, phi, action, reward):
        phi = np.asarray(phi, dtype=np.float32).ravel()
        a = np.atleast_1d(np.asarray(action, dtype=np.float32)).ravel()
        r = np.asarray([math.sqrt(len(phi)) * float(reward)], dtype=np.float32)
        return np.concatenate([phi, a, r], axis=0)

    def _on_detected(self):
        self.detections.append(self.ts)
        self.last_shift_step = self.ts
        # Purge buffers so the new regime is not polluted by old statistics.
        self._buffer.clear()
        self._swd_hist.clear()
        # Reset the cached p-value so a stale post-fire value doesn't trip a
        # downstream aggregator (e.g. HybridDetector) on the next tick.
        self.last_pval = 1.0
        self.last_swd = 0.0


# ----------------------------------------------------------------------
# Evaluation utilities
# ----------------------------------------------------------------------
def match_detections(detected_steps, oracle_steps, tolerance):
    """Greedy one-to-one matching between detected and oracle switch times.

    A detection at step d matches an oracle switch at step o iff
        o <= d <= o + tolerance
    (detections are always after the true switch because the detector needs
    to accumulate post-switch samples before it can fire).  Returns a dict
    with TP, FP, FN, per-oracle detection delays, and F1.
    """
    oracle_steps = sorted(oracle_steps)
    detected_steps = sorted(detected_steps)
    matched = [False] * len(oracle_steps)
    delays, tps = [], []
    fps = []

    for d in detected_steps:
        paired = False
        for i, o in enumerate(oracle_steps):
            if matched[i]:
                continue
            if o <= d <= o + tolerance:
                matched[i] = True
                paired = True
                tps.append((o, d))
                delays.append(d - o)
                break
        if not paired:
            fps.append(d)

    tp = len(tps)
    fp = len(fps)
    fn = sum(1 for m in matched if not m)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mean_delay": float(np.mean(delays)) if delays else float("nan"),
        "max_delay": int(max(delays)) if delays else -1,
        "tps": tps,
        "fps": fps,
    }
