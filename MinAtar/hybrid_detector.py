"""Hybrid task-shift detector for FAME (Approach 3 in the proposal).

Why *not* a simple intersection
-------------------------------
The proposal's hybrid rule is
    Shift_hyb = Suspect * 1[p_stat < alpha]
i.e. an AND gate.  The obvious problem: two independent boundary detectors,
each tuned for its own FP rate, applied as an AND produce a VERY conservative
combined detector with poor recall.  Worse, an AND gate wastes information:
if one detector is screaming (p = 1e-12) and the other is mildly confident
(p = 0.2), the AND fails even though the joint evidence is overwhelming.

Our rigid replacement: a 3-state cascade with log-evidence fusion
-----------------------------------------------------------------
We design a small finite-state machine over two p-values (p_imp from
`ImplicitDetector`, p_stat from `SwoksDetector`) that captures the sequential
nature of the two signals (implicit = fast reactive, statistical = slower
and more reliable) while preserving a Bayesian-style evidence fusion.

    States: NEUTRAL  (no evidence of shift)
            SUSPECT  (implicit raised a fast flag; snapshot captured)
            CONFIRMED(fire) -- transient, reverts to NEUTRAL after firing

Transitions (evaluated at every detection step, synchronised across both
inner detectors):

    NEUTRAL -> SUSPECT
        when implicit p_imp < tau_imp_loose   for `persistence` consecutive
        ticks  (persistence > 1 kills single-tick noise)
        On entry: snap_step <- t;  external snapshot hook fires

    SUSPECT -> CONFIRMED (fire)
        triggered by ANY of the following, whichever happens first:
        (a) p_stat < tau_stat_strict                       (SWOKS confirms)
        (b) p_imp  < tau_imp_strict                        (implicit extreme)
        (c) -log(p_imp) + -log(p_stat) > tau_combined      (joint evidence)

    SUSPECT -> NEUTRAL (timeout / recovery)
        if t - snap_step > horizon                         (suspicion expires)
        OR implicit recovers (p_imp > 0.5)                 (signal vanished)

Failsafes (parallel to the FSM):
    * **Pure-statistical failsafe**: if p_stat < 1e-6 while in NEUTRAL, fire
      immediately.  Catches the rare case where reward/dynamics don't change
      much but the latent action-reward distribution clearly shifts.
    * **Pure-implicit failsafe**: if p_imp < tau_imp_strict for q=3
      consecutive ticks while in NEUTRAL (didn't trip SUSPECT for some
      reason), fire immediately.

Mathematical justification of the combined rule
------------------------------------------------
Under the null H0 (no shift), p_imp and p_stat are approximately Uniform[0,1]
and approximately independent because the two detectors operate on different
features (prediction error vs. latent distribution distance).  Therefore
-log(p_imp) and -log(p_stat) are Exp(1)-distributed, and their sum is
Gamma(shape=2, scale=1) with CDF F(x) = 1 - (1+x)e^{-x}.  A threshold of
tau_combined = 15 (nats) corresponds to joint FP rate ~ 5e-6, a very tight
combined gate.  During SUSPECT we also allow a single-detector bypass, so
the *effective* FP rate sits between the two single-detector rates.

Snapshot protection
-------------------
FAME needs a pre-contamination fast learner as a warm-up candidate.  The
live fast learner has been updating on post-shift data since the first
implicit warning (potentially hundreds or thousands of steps ago), so its
weights are polluted.  Taking a snapshot at the moment of SUSPECT entry
gives us a clean pre-contamination copy.  The FAME integration layer uses
this snapshot as `fast_candidate` in the warm-up hypothesis test.

Interface
---------
    det = HybridDetector(implicit_det, swoks_det, ...)
    fired = det.step(phi, action, reward, next_phi)  # delegates to both
    det.snapshot_requested          # bool, set on SUSPECT entry, cleared on read
    det.state                       # 'NEUTRAL' | 'SUSPECT'
    det.stats()
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Callable, Optional


_NEUTRAL = "NEUTRAL"
_SUSPECT = "SUSPECT"
# (no CONFIRMED literal state -- firing is transient)


class HybridDetector:
    """Sequential cascade + log-evidence fusion over two base detectors.

    Parameters
    ----------
    implicit : ImplicitDetector
        Fast, reactive detector -- its p-value drives SUSPECT entry.
    statistical : SwoksDetector
        Slow, reliable detector -- its p-value drives CONFIRMED entry.
    tau_imp_loose : float
        Upper p-value for SUSPECT entry (e.g. 1e-2).
    tau_imp_strict : float
        Upper p-value for single-detector bypass via implicit (e.g. 1e-6).
    tau_stat_strict : float
        Upper p-value for single-detector bypass via SWOKS (e.g. 1e-3).
    tau_stat_failsafe : float
        Upper p-value for pure-statistical failsafe from NEUTRAL (e.g. 1e-6).
    tau_combined : float
        Joint evidence threshold in *nats* (log(1/p_imp) + log(1/p_stat)).
    horizon : int
        Max steps SUSPECT remains active before timing out back to NEUTRAL.
    persistence : int
        Consecutive NEUTRAL-level implicit ticks required to enter SUSPECT.
    cooldown : int or None
        Minimum steps between two consecutive fires (aligns with the inner
        detectors' stable_phase semantics).  Defaults to the implicit
        detector's stable_phase.
    on_suspect : callable or None
        Optional hook invoked at SUSPECT entry; typically used to stash a
        pre-contamination fast-learner snapshot.
    """

    def __init__(
        self,
        implicit,
        statistical,
        *,
        tau_imp_loose: float = 1e-2,
        tau_imp_strict: float = 1e-6,
        tau_stat_strict: float = 1e-3,
        tau_stat_failsafe: float = 1e-6,
        tau_combined: float = 15.0,
        horizon: int = 480,
        persistence: int = 2,
        cooldown: Optional[int] = None,
        on_suspect: Optional[Callable[[int], Any]] = None,
    ):
        self.imp = implicit
        self.stat = statistical
        self.tau_imp_loose = float(tau_imp_loose)
        self.tau_imp_strict = float(tau_imp_strict)
        self.tau_stat_strict = float(tau_stat_strict)
        self.tau_stat_failsafe = float(tau_stat_failsafe)
        self.tau_combined = float(tau_combined)
        self.horizon = int(horizon)
        self.persistence = int(persistence)
        # Default cooldown to the implicit detector's stable_phase.  This
        # mirrors SWOKS' behaviour of suppressing detection for stable_phase
        # steps after every fire -- a hybrid without this safeguard would
        # re-fire on the same shift for as long as its inner p-values remain
        # below threshold.
        self.cooldown = int(cooldown) if cooldown is not None \
            else int(getattr(implicit, "stable_phase", 0))
        self.on_suspect = on_suspect

        self.state = _NEUTRAL
        self.ts = 0
        self.last_fire_ts = -10**9
        self.suspect_entered_at = None
        self._loose_streak = 0
        self._strict_streak = 0

        # Bookkeeping for introspection.
        self.detections = []        # list of absolute time steps
        self.events = []            # list of (ts, state, reason)
        self.snapshot_requested = False
        self.last_p_imp = 1.0
        self.last_p_stat = 1.0
        self.last_combined = 0.0
        self.last_reason = ""

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------
    def step(self, phi, action, reward, next_phi) -> bool:
        self.ts += 1

        # Feed both inner detectors.  We do NOT act on their own `fired`
        # signals: the hybrid state machine decides when to fire.  This
        # means the inner detectors' internal `last_shift_step` is NOT
        # advanced, so their stable_phase gating is inactive for us.
        imp_fired = self.imp.step(phi, action, reward, next_phi)
        stat_fired = self.stat.step(phi, action, reward)

        # Respect cooldown: don't re-evaluate firing rules for `cooldown`
        # steps after a confirmed shift.  This prevents re-firing on the
        # same shift while the inner detectors re-fill their windows.
        if self.ts - self.last_fire_ts < self.cooldown:
            return False
        # (We intentionally ignore imp_fired/stat_fired except for the
        # pure-failsafe checks below.  The inner detectors will still
        # advance their internal counters, which is fine: once WE fire,
        # we'll call reset_after_boundary() on both to clear them.)

        p_imp = self.imp.last_pval
        p_stat = self.stat.last_pval
        self.last_p_imp = float(p_imp)
        self.last_p_stat = float(p_stat)
        self.last_combined = float(
            -math.log(max(p_imp, 1e-300)) - math.log(max(p_stat, 1e-300))
        )

        fired = False
        reason = ""

        # ------- pure-detector failsafes (active only in NEUTRAL) ---------
        # These compare directly against the live p-values, so they work
        # regardless of whether the inner detectors are configured with
        # their own firing thresholds enabled (they typically aren't when
        # operating as sub-detectors of the hybrid).
        if self.state == _NEUTRAL:
            if p_stat < self.tau_stat_failsafe:
                fired = True
                reason = "pure-stat"
            elif p_imp < self.tau_imp_strict:
                # Track a streak of extreme implicit evidence so that one
                # noisy tick alone doesn't bypass the SUSPECT stage.
                self._strict_streak += 1
                if self._strict_streak >= self.persistence:
                    fired = True
                    reason = "pure-imp-strict"
            else:
                self._strict_streak = 0

        # ------- FSM transitions ----------------------------------------
        if not fired:
            if self.state == _NEUTRAL:
                if p_imp < self.tau_imp_loose:
                    self._loose_streak += 1
                else:
                    self._loose_streak = 0
                if self._loose_streak >= self.persistence:
                    self._enter_suspect()

            elif self.state == _SUSPECT:
                # Confirmation paths
                if p_stat < self.tau_stat_strict:
                    fired = True
                    reason = "suspect->stat-strict"
                elif p_imp < self.tau_imp_strict:
                    fired = True
                    reason = "suspect->imp-strict"
                elif self.last_combined > self.tau_combined:
                    fired = True
                    reason = "suspect->combined"
                elif self.ts - self.suspect_entered_at > self.horizon:
                    self._revert_to_neutral("horizon-timeout")
                elif p_imp > 0.5:
                    self._revert_to_neutral("imp-recovered")

        if fired:
            self._on_fired(reason)
        return fired

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    def _enter_suspect(self):
        self.state = _SUSPECT
        self.suspect_entered_at = self.ts
        self.snapshot_requested = True
        self.events.append((self.ts, "SUSPECT", "enter"))
        if self.on_suspect is not None:
            try:
                self.on_suspect(self.ts)
            except Exception:
                pass

    def _revert_to_neutral(self, reason: str):
        self.state = _NEUTRAL
        self.suspect_entered_at = None
        self._loose_streak = 0
        self.events.append((self.ts, "NEUTRAL", f"revert:{reason}"))

    def _on_fired(self, reason: str):
        self.detections.append(self.ts)
        self.last_fire_ts = self.ts
        self.last_reason = reason
        self.events.append((self.ts, "FIRE", reason))
        # Clear inner detectors so their stable_phase gating protects us too.
        try:
            self.imp.reset_after_boundary()
            self.stat.reset_after_boundary()
        except Exception:
            pass
        # Return to NEUTRAL (consume any pending snapshot request).
        self.state = _NEUTRAL
        self.suspect_entered_at = None
        self._loose_streak = 0
        self._strict_streak = 0
        self.snapshot_requested = False

    # ------------------------------------------------------------------
    # External API
    # ------------------------------------------------------------------
    def consume_snapshot_request(self) -> bool:
        """Read-and-clear the snapshot flag.  Caller stashes a fast-learner
        snapshot when this returns True."""
        if self.snapshot_requested:
            self.snapshot_requested = False
            return True
        return False

    def reset_after_boundary(self):
        """External reset (e.g. when the oracle fires in debug mode)."""
        self._on_fired("external")

    def stats(self) -> dict:
        return {
            "ts": self.ts,
            "state": self.state,
            "suspect_entered_at": self.suspect_entered_at,
            "num_detections": len(self.detections),
            "last_p_imp": self.last_p_imp,
            "last_p_stat": self.last_p_stat,
            "last_combined_nats": self.last_combined,
            "last_reason": self.last_reason,
            "num_events": len(self.events),
            "inner_imp": self.imp.stats(),
            "inner_stat": self.stat.stats(),
        }


__all__ = ["HybridDetector"]
