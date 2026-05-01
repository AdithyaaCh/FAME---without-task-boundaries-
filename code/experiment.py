from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Plotting is optional at import-time -- the aggregate path still works
# even if matplotlib is missing (useful on headless CI machines).
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_theme(context="paper", style="whitegrid",
                      rc={"axes.spines.right": False,
                          "axes.spines.top": False})
        _HAS_SEABORN = True
    except Exception:
        _HAS_SEABORN = False
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False
    _HAS_SEABORN = False

try:
    import torch
    from model import CNN
    from CL_envs import CL_envs_func_replacement
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
DETECTOR_MODES: Tuple[str, ...] = ("oracle", "swoks", "implicit", "hybrid")
BASELINE_MODE = "oracle"  # for FT computations against a "Reset"-like ceiling
GAMES: Tuple[str, ...] = ("breakout", "space_invaders", "freeway")
GAME_ID = {g: i for i, g in enumerate(GAMES)}


# ======================================================================
# 1. Data loading
# ======================================================================
def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def discover_runs(results_dir: str,
                  seq: int,
                  seeds: Sequence[int],
                  modes: Sequence[str] = DETECTOR_MODES) -> Dict[str, Dict[int, str]]:
    """Map {mode -> {seed -> path}} for FAME_*_returns.pkl files."""
    out: Dict[str, Dict[int, str]] = {m: {} for m in modes}
    for mode in modes:
        pattern = os.path.join(
            results_dir,
            f"FAME_{mode}_*seq_{seq}*seed_*_returns.pkl"
        )
        for p in glob.glob(pattern):
            m = re.search(rf"seed_(\d+)_returns\.pkl$", p)
            if not m:
                continue
            s = int(m.group(1))
            if s not in seeds:
                continue
            # If multiple pickles match the same (mode, seed) -- e.g. stale
            # runs with a different step count left over from earlier
            # experimentation -- keep the most recently modified one.  This
            # guarantees experiment.py analyses the freshest artifact and
            # that its MetaFinal weights are the ones that correspond to it.
            prev = out[mode].get(s)
            if prev is None or os.path.getmtime(p) > os.path.getmtime(prev):
                out[mode][s] = p
    return out


# ======================================================================
# 2. Core training-trace metrics (always computable)
# ======================================================================
def smooth(x: np.ndarray, window: int = 2000) -> np.ndarray:
    """Trailing moving average for plotting."""
    if window <= 1 or len(x) <= 1:
        return x.astype(np.float64)
    w = min(window, len(x))
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(x.astype(np.float64), kernel, mode="same")


def per_task_auc_normalized(returns: np.ndarray,
                            boundaries: Sequence[int],
                            total_steps: int) -> Tuple[List[float], List[Tuple[float, float]]]:
    """AUC_i per task after per-task min-max normalisation to [0, 1].

    Paper's definition normalises to ensure AUC_i in [0,1].  We do it
    per-task using the per-task min/max over the training trace, matching
    the paper's footnote convention for pixel-based tasks.
    """
    edges = [0] + list(boundaries) + [total_steps]
    auc, norm_bounds = [], []
    for s, e in zip(edges[:-1], edges[1:]):
        seg = returns[s:e]
        if len(seg) == 0:
            continue
        lo = float(np.min(seg))
        hi = float(np.max(seg))
        if hi - lo < 1e-9:
            auc.append(0.0)
        else:
            auc.append(float((seg - lo).mean() / (hi - lo)))
        norm_bounds.append((lo, hi))
    return auc, norm_bounds


def forward_transfer(auc_curr: Sequence[float],
                     auc_base: Sequence[float]) -> List[float]:
    """Per-task FT_i = (AUC_i - AUC_b_i) / (1 - AUC_b_i); paper Eq. 9."""
    out = []
    for c, b in zip(auc_curr, auc_base):
        denom = 1.0 - b
        out.append((c - b) / denom if abs(denom) > 1e-6 else 0.0)
    return out


def avg_performance_proxy(returns: np.ndarray, tail_frac: float = 0.02) -> float:
    """Mean over the last `tail_frac` of the training trace."""
    n = len(returns)
    k = max(1, int(round(n * tail_frac)))
    return float(np.asarray(returns[-k:]).mean())


def forgetting_proxy(returns: np.ndarray,
                     boundaries: Sequence[int],
                     total_steps: int,
                     window: int = 5000) -> Tuple[float, List[float]]:
    """Training-trace forgetting proxy: (end-of-task mean) - (end-of-run mean).

    KNOWN LIMITATIONS (see per_game_metrics for fixes):
    - end-of-run is the EMA on whatever game is *last* in the sequence; for
      tasks of a different game this is comparing apples to oranges.
    - methods that correctly identify same-game transitions (e.g. SWOKS on
      seq=0 task 4->5, both space_invaders) accumulate higher peaks and
      therefore score higher "forgetting" values purely because they had
      more skill to lose.  The metric conflates "learned a lot" with
      "forgot a lot".

    Kept here for backward compatibility with the FAME paper's reported
    proxy.  For analysis we now also emit per_game_metrics() below, which
    properly evaluates each game's last training window separately.
    """
    edges = list(boundaries) + [total_steps]
    end_win = returns[max(0, total_steps - window):total_steps].mean()
    per_task = []
    for e in edges[:-1]:
        e = int(e)
        win = returns[max(0, e - window):e].mean()
        per_task.append(float(win - end_win))
    agg = float(np.mean(per_task)) if per_task else 0.0
    return agg, per_task


def per_game_metrics(returns: np.ndarray,
                     boundaries: Sequence[int],
                     games: Sequence[str],
                     total_steps: int,
                     window: int = 5000) -> dict:
    """Game-aware training-trace metrics.

    Three metrics, all computed per unique game in the sequence:

    * `last_window`   : mean EMA return in the last `window` steps of the
                        LAST task that was this game.  Best training-trace
                        proxy for "current skill on game g".
    * `peak_window`   : max over all tasks of game g of the end-of-task
                        window mean.  "Best the method ever got on g".
    * `retention`     : last_window / peak_window  (clipped to [0, inf]).
                        Bounded ~[0, 1] under normal training; > 1 if the
                        last occurrence somehow exceeds the historical
                        peak.  This is the right way to talk about
                        forgetting because it doesn't penalise methods
                        that simply learned more (the original
                        forgetting_proxy does).

    Aggregate averages across games are also returned.

    Why this is better than avg_perf_proxy and forgetting_proxy:
    - Equal weight per game, not per task.  seq=0 has 4 tasks of breakout
      (tasks 1, 3, 7) and 3 of space_invaders (2, 4, 5) and 1 of freeway
      (6).  avg_perf_proxy weights them by sequence position; this
      weights them by uniqueness.
    - Uses the most recent training window for each game, so it answers
      the question "after the full curriculum, how good is the method at
      each game?" without any post-hoc rollouts.
    - retention is bounded and meaningful: 1.0 means "you still have
      everything you ever learned"; 0.0 means "complete forgetting".
    """
    edges = [0] + list(boundaries) + [total_steps]
    # task_windows[task_idx] = (start, end, end_win_mean, max_in_task)
    task_info = []
    for i, (s, e) in enumerate(zip(edges[:-1], edges[1:])):
        e = int(e); s = int(s)
        if e <= s:
            continue
        end_win = float(returns[max(s, e - window):e].mean())
        task_info.append((i, s, e, end_win))

    by_game = {}  # game -> list of (task_idx, end_win)
    for (i, s, e, ew) in task_info:
        if i >= len(games):
            continue
        g = games[i]
        by_game.setdefault(g, []).append((i, ew))

    # Retention threshold: if a game's peak end-of-task return is below this
    # the method never really learned that game, so retention is undefined
    # (you can't "forget" what you never knew).  Without this, the freeway
    # game on seq=0 -- which most methods score ~0 on -- would contribute a
    # spurious retention=1.0 to every method's average and drag the metric
    # toward meaninglessness.
    MIN_PEAK_FOR_RETENTION = 1.0

    per_game_last = {}    # most-recent-occurrence end window per game
    per_game_peak = {}    # max end window across all occurrences of game
    per_game_retention = {}    # only populated for games with peak >= MIN
    for g, occs in by_game.items():
        if not occs:
            continue
        last_ew = occs[-1][1]
        peak_ew = max(o[1] for o in occs)
        per_game_last[g] = last_ew
        per_game_peak[g] = peak_ew
        if peak_ew >= MIN_PEAK_FOR_RETENTION:
            per_game_retention[g] = last_ew / peak_ew

    avg_last = float(np.mean(list(per_game_last.values()))) if per_game_last else 0.0
    avg_peak = float(np.mean(list(per_game_peak.values()))) if per_game_peak else 0.0
    # Retention is averaged only over games the method actually learned.
    # If no game was learned, retention is undefined (returned as nan).
    if per_game_retention:
        avg_retention = float(np.mean(list(per_game_retention.values())))
    else:
        avg_retention = float("nan")

    return {
        "per_game_last": per_game_last,
        "per_game_peak": per_game_peak,
        "per_game_retention": per_game_retention,
        "avg_last_per_game": avg_last,
        "avg_peak_per_game": avg_peak,
        "avg_retention": avg_retention,
    }


# ======================================================================
# 3. Detection metrics
# ======================================================================
def match_detections(detected: Sequence[int],
                     oracle: Sequence[int],
                     tolerance: int) -> dict:
    """Greedy 1-to-1 matching of detected vs oracle switches within tolerance.

    A detection d matches an oracle switch o iff o <= d <= o + tolerance.
    Returns TP/FP/FN/precision/recall/F1/mean & max delay.
    """
    oracle = sorted(oracle)
    detected = sorted(detected)
    matched = [False] * len(oracle)
    tps, fps, delays = [], [], []
    for d in detected:
        paired = False
        for i, o in enumerate(oracle):
            if matched[i]:
                continue
            if o <= d <= o + tolerance:
                matched[i] = True
                tps.append((o, d))
                delays.append(d - o)
                paired = True
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
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "mean_delay": float(np.mean(delays)) if delays else float("nan"),
        "max_delay": int(max(delays)) if delays else -1,
        "tps": tps, "fps": fps,
        "delays": delays,
    }


def tolerance_sweep(detected: Sequence[int],
                    oracle: Sequence[int],
                    tols: Sequence[int]) -> List[dict]:
    """F1 / TP / FP curves at a range of tolerances."""
    return [{"tolerance": t, **match_detections(detected, oracle, t)}
            for t in tols]


# ======================================================================
# 4. Post-hoc policy evaluation (paper-strict p_i(K*T))
# ======================================================================
def _infer_obs_shape_and_actions(game: str = "breakout", seed: int = 0):
    env = CL_envs_func_replacement(seq=0, game_id=GAME_ID[game], seed=seed,
                                   evaluation=True)
    s = env.reset(seed=seed)
    return s.shape, env.action_space.n


def _obs_to_tensor(obs, device):
    obs = np.moveaxis(obs, 2, 0)
    return torch.tensor(obs, dtype=torch.float, device=device).unsqueeze(0)


def evaluate_policy_in_env(model: "CNN",
                           game: str,
                           seed: int,
                           n_episodes: int,
                           max_steps_per_ep: int,
                           device: str) -> Tuple[float, float]:
    """Greedy rollout of `model` in `game`; returns (mean_return, std_return)."""
    env = CL_envs_func_replacement(seq=0, game_id=GAME_ID[game], seed=seed,
                                   evaluation=True)
    returns = []
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        ep_ret, done = 0.0, False
        for _ in range(max_steps_per_ep):
            with torch.no_grad():
                q = model(_obs_to_tensor(obs, device))
            a = int(q.argmax(dim=1).item())
            obs, r, done, _ = env.step(a)
            ep_ret += float(r)
            if done:
                break
        returns.append(ep_ret)
    arr = np.asarray(returns, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def posthoc_evaluate(models_dir: str,
                     run_filename_stem: str,
                     games: Sequence[str] = GAMES,
                     n_episodes: int = 20,
                     max_steps_per_ep: int = 2000,
                     device: str = "cpu",
                     in_channels: int = 7,
                     num_actions: int = 6,
                     eval_seed: int = 9999) -> Dict[str, Tuple[float, float]]:
    """Load {stem}_MetaFinal.pt and evaluate across `games`.

    Returns {game_name: (mean_return, std_return)}.
    """
    if not _HAS_TORCH:
        raise RuntimeError("Post-hoc eval requires torch -- install FAMEenv.")
    meta_path = os.path.join(models_dir, f"{run_filename_stem}_MetaFinal.pt")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"missing final meta weights: {meta_path}")
    model = CNN(in_channels, num_actions).to(device)
    sd = torch.load(meta_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    out = {}
    for g in games:
        mu, sd = evaluate_policy_in_env(
            model, g, eval_seed, n_episodes, max_steps_per_ep, device
        )
        out[g] = (mu, sd)
    return out


# ======================================================================
# 5. Aggregation
# ======================================================================
@dataclass
class RunSummary:
    mode: str
    seed: int
    seq: int
    n_steps: int
    games: List[str]
    oracle_boundaries: List[int]
    detected_boundaries: List[int]
    flag_history: List[str]
    detection_log: List[dict]
    auc_per_task: List[float]
    ft_per_task: List[float]                 # filled by aggregation vs baseline
    avg_perf_proxy: float
    forgetting_proxy_total: float
    forgetting_proxy_per_task: List[float]
    # Game-aware training-trace metrics (proper aggregate; see per_game_metrics)
    avg_perf_per_game: float = 0.0      # mean across games of last-occurrence window
    avg_retention: float = 1.0          # mean across games of last_window / peak_window
    per_game_last: Dict[str, float] = field(default_factory=dict)
    per_game_peak: Dict[str, float] = field(default_factory=dict)
    per_game_retention: Dict[str, float] = field(default_factory=dict)
    detection: dict = field(default_factory=dict)
    # Post-hoc fields (may be None if not evaluated)
    posthoc_avg_perf: Optional[float] = None
    posthoc_per_game: Optional[Dict[str, Tuple[float, float]]] = None
    # Misc
    hybrid_reasons: Counter = field(default_factory=Counter)
    filepath: str = ""
    filename_stem: str = ""


def run_filename_stem(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith("_returns.pkl"):
        return base[: -len("_returns.pkl")]
    return base.rsplit(".", 1)[0]


def summarise_run(path: str, *, tolerance: int) -> RunSummary:
    data = load_pickle(path)
    returns = np.asarray(data["returns"])
    oracle = sorted(data.get("oracle_boundaries", []))
    detected = sorted(data.get("detected_boundaries", []))
    games = list(data.get("games", []))
    flag_history = list(data.get("flag_history", []))
    detection_log = list(data.get("detection_log", []))
    args = data.get("args", {})
    seq = int(args.get("seq", -1))
    seed = int(args.get("seed", -1))
    mode = data.get("mode", args.get("detector", "?"))

    auc, _ = per_task_auc_normalized(returns, oracle, len(returns))
    avg_p = avg_performance_proxy(returns, tail_frac=0.02)
    forg_total, forg_per = forgetting_proxy(returns, oracle, len(returns))
    pg = per_game_metrics(returns, oracle, games, len(returns))
    det = match_detections(detected, oracle, tolerance=tolerance)

    reasons = Counter(
        str(e.get("reason", "")) for e in detection_log if e.get("reason")
    )

    return RunSummary(
        mode=mode, seed=seed, seq=seq, n_steps=len(returns),
        games=games, oracle_boundaries=oracle,
        detected_boundaries=detected, flag_history=flag_history,
        detection_log=detection_log,
        auc_per_task=auc,
        ft_per_task=[],  # filled later
        avg_perf_proxy=avg_p,
        forgetting_proxy_total=forg_total,
        forgetting_proxy_per_task=forg_per,
        avg_perf_per_game=pg["avg_last_per_game"],
        avg_retention=pg["avg_retention"],
        per_game_last=pg["per_game_last"],
        per_game_peak=pg["per_game_peak"],
        per_game_retention=pg["per_game_retention"],
        detection=det,
        hybrid_reasons=reasons,
        filepath=path,
        filename_stem=run_filename_stem(path),
    )


def compute_forward_transfer(summaries_by_mode: Dict[str, List[RunSummary]],
                             baseline: str = BASELINE_MODE) -> None:
    """Populate rs.ft_per_task on every summary using the baseline mode.

    Per-task baseline is the mean AUC across the baseline mode's seeds.  If
    the baseline mode isn't present we fall back to zero AUC (=raw FT).
    """
    if baseline in summaries_by_mode and summaries_by_mode[baseline]:
        # Average per-task across seeds, padded to the same length.
        lens = [len(r.auc_per_task) for r in summaries_by_mode[baseline]]
        L = min(lens)
        stack = np.stack([np.asarray(r.auc_per_task[:L])
                          for r in summaries_by_mode[baseline]])
        base_auc = stack.mean(axis=0).tolist()
    else:
        base_auc = None
    for mode, runs in summaries_by_mode.items():
        for rs in runs:
            if base_auc is None:
                rs.ft_per_task = [a for a in rs.auc_per_task]  # = raw AUC
            else:
                L = min(len(base_auc), len(rs.auc_per_task))
                rs.ft_per_task = forward_transfer(rs.auc_per_task[:L],
                                                  base_auc[:L])


def normalise_forgetting_across_methods(summaries_by_mode: Dict[str, List[RunSummary]]
                                        ) -> Dict[str, List[float]]:
    """FAME paper: per-task forgetting normalised by cross-method std.

    Returns {mode -> list[per-task-normalised-F]}.
    """
    modes = list(summaries_by_mode.keys())
    if not modes:
        return {}
    max_tasks = max(
        max((len(r.forgetting_proxy_per_task) for r in runs), default=0)
        for runs in summaries_by_mode.values()
    )
    if max_tasks == 0:
        return {m: [] for m in modes}

    # Stack: shape (modes x seeds x tasks), padding tasks to max.
    per_task_values: Dict[str, np.ndarray] = {}
    for m, runs in summaries_by_mode.items():
        if not runs:
            per_task_values[m] = np.full(max_tasks, np.nan)
            continue
        stack = np.full((len(runs), max_tasks), np.nan)
        for i, r in enumerate(runs):
            v = r.forgetting_proxy_per_task[:max_tasks]
            stack[i, :len(v)] = v
        # Seed-mean.
        per_task_values[m] = np.nanmean(stack, axis=0)
    # Cross-method std per task.
    matrix = np.vstack([per_task_values[m] for m in modes])
    stds = np.nanstd(matrix, axis=0, ddof=0)
    stds[stds < 1e-9] = 1.0  # guard
    return {m: (per_task_values[m] / stds).tolist() for m in modes}


def mean_se(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    mu = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mu, se


# ======================================================================
# 6. Visualisations
# ======================================================================
def _ensure_plots_dir(out_dir: str) -> str:
    d = os.path.join(out_dir, "plots")
    os.makedirs(d, exist_ok=True)
    return d


def _has_mpl() -> bool:
    if not _HAS_MPL:
        print("[plots] matplotlib not available, skipping figures.")
    return _HAS_MPL


def _thin(arr: np.ndarray, max_pts: int = 4000) -> np.ndarray:
    """Stride-downsample *arr* to at most *max_pts* points.

    The AGG renderer raises OverflowError: Exceeded cell block limit when asked
    to render more than ~1M path segments.  With 3.5M training steps per run,
    learning-curve arrays must be thinned before plotting.  A stride of
    ceil(N/max_pts) preserves the visual shape of a heavily smoothed curve
    while keeping the point count well inside the renderer's limit.
    """
    n = len(arr)
    if n <= max_pts:
        return arr
    step = math.ceil(n / max_pts)
    return arr[::step]


def plot_learning_curves(summaries_by_mode: Dict[str, List[RunSummary]],
                         out_path: str,
                         smooth_window: int = 3000) -> None:
    if not _has_mpl():
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode, runs in summaries_by_mode.items():
        if not runs:
            continue
        # Align to the shortest run in case of small mismatches.
        L = min(r.n_steps for r in runs)
        stack = np.stack([
            smooth(np.asarray(load_pickle(r.filepath)["returns"])[:L],
                   window=smooth_window)
            for r in runs
        ])
        mu = stack.mean(axis=0)
        se = stack.std(axis=0, ddof=0) / math.sqrt(stack.shape[0])
        # Downsample before plotting: 3.5M points triggers AGG cell-block overflow.
        xs = _thin(np.arange(L))
        mu_plot = _thin(mu)
        se_plot = _thin(se)
        ax.plot(xs, mu_plot, label=mode, linewidth=1.4)
        ax.fill_between(xs, mu_plot - se_plot, mu_plot + se_plot, alpha=0.15)
    # Oracle vertical lines -- pick first run that has them.
    for runs in summaries_by_mode.values():
        if runs:
            for b in runs[0].oracle_boundaries:
                ax.axvline(b, color="k", alpha=0.15, linewidth=0.6)
            break
    ax.set_xlabel("Training step")
    ax.set_ylabel(f"Return (smoothed w={smooth_window})")
    ax.set_title("Learning curves: mean +/- SE across seeds")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_detection_timeline(runs_by_mode: Dict[str, List[RunSummary]],
                            out_path: str) -> None:
    if not _has_mpl():
        return
    modes = [m for m, rs in runs_by_mode.items() if rs and m != "oracle"]
    if not modes:
        return
    # Use the first seed of each mode for compactness.
    seeds_common = sorted(set.intersection(
        *[{r.seed for r in runs_by_mode[m]} for m in modes]
    ))
    if not seeds_common:
        return
    seed = seeds_common[0]
    picks = {m: next(r for r in runs_by_mode[m] if r.seed == seed)
             for m in modes}

    fig, axes = plt.subplots(len(modes), 1,
                             figsize=(9, 1.6 * len(modes) + 0.5),
                             sharex=True)
    if len(modes) == 1:
        axes = [axes]
    oracle = picks[modes[0]].oracle_boundaries

    for ax, m in zip(axes, modes):
        rs = picks[m]
        # Oracle boundaries as grey bands.
        for o in oracle:
            ax.axvline(o, color="grey", linestyle="--", linewidth=1,
                       alpha=0.8)
        # Detections as vertical stems, coloured by reason (hybrid) or mode.
        colors = {
            "suspect->imp-strict": "#1f77b4",
            "suspect->stat-strict": "#2ca02c",
            "suspect->combined": "#9467bd",
            "pure-imp-strict": "#d62728",
            "pure-stat": "#ff7f0e",
        }
        for entry in rs.detection_log:
            step = entry.get("step", 0)
            reason = entry.get("reason", m)
            c = colors.get(reason, "#333333")
            ax.axvline(step, color=c, linewidth=1.2, alpha=0.9)
        ax.set_ylabel(m, rotation=0, ha="right", va="center", labelpad=20)
        ax.set_yticks([])
        ax.set_xlim(0, rs.n_steps)
    axes[-1].set_xlabel("Step")
    fig.suptitle(f"Detection timeline (seed={seed}, grey=oracle, coloured=detected)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_detection_delay_hist(runs_by_mode: Dict[str, List[RunSummary]],
                              out_path: str) -> None:
    if not _has_mpl():
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for mode, runs in runs_by_mode.items():
        if mode == "oracle" or not runs:
            continue
        all_delays = []
        for r in runs:
            all_delays.extend(r.detection["delays"])
        if not all_delays:
            continue
        ax.hist(all_delays, bins=20, alpha=0.5, label=f"{mode} (n={len(all_delays)})")
    ax.set_xlabel("Detection delay (steps)")
    ax.set_ylabel("Count (TPs)")
    ax.set_title("Distribution of detection delays (TPs across seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_detection_quality(runs_by_mode: Dict[str, List[RunSummary]],
                           out_path: str) -> None:
    if not _has_mpl():
        return
    modes = [m for m, rs in runs_by_mode.items() if rs and m != "oracle"]
    if not modes:
        return
    tp = [sum(r.detection["tp"] for r in runs_by_mode[m]) for m in modes]
    fp = [sum(r.detection["fp"] for r in runs_by_mode[m]) for m in modes]
    fn = [sum(r.detection["fn"] for r in runs_by_mode[m]) for m in modes]
    f1 = [np.mean([r.detection["f1"] for r in runs_by_mode[m]]) for m in modes]

    x = np.arange(len(modes))
    w = 0.25
    fig, ax1 = plt.subplots(figsize=(6.5, 3.8))
    ax1.bar(x - w, tp, width=w, label="TP", color="#2ca02c")
    ax1.bar(x, fp, width=w, label="FP", color="#d62728")
    ax1.bar(x + w, fn, width=w, label="FN", color="#7f7f7f")
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes)
    ax1.set_ylabel("Count")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, f1, marker="o", color="black", label="F1")
    for xi, fi in zip(x, f1):
        ax2.text(xi, fi + 0.02, f"{fi:.2f}", ha="center", fontsize=8)
    ax2.set_ylabel("Mean F1")
    ax2.set_ylim(0, 1.05)

    ax1.set_title("Detection quality across modes (pooled over seeds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_per_task_auc(summaries_by_mode: Dict[str, List[RunSummary]],
                      out_path: str) -> None:
    if not _has_mpl():
        return
    modes = [m for m, rs in summaries_by_mode.items() if rs]
    if not modes:
        return
    max_tasks = max(max(len(r.auc_per_task) for r in summaries_by_mode[m])
                    for m in modes)
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.8 / max(1, len(modes))
    x = np.arange(max_tasks)
    for i, m in enumerate(modes):
        stack = np.full((len(summaries_by_mode[m]), max_tasks), np.nan)
        for j, r in enumerate(summaries_by_mode[m]):
            stack[j, :len(r.auc_per_task)] = r.auc_per_task
        mu = np.nanmean(stack, axis=0)
        se = np.nanstd(stack, axis=0, ddof=0) / math.sqrt(max(1, len(summaries_by_mode[m])))
        ax.bar(x + (i - len(modes) / 2) * w + w / 2, mu, width=w,
               yerr=se, capsize=2, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([f"task {i+1}" for i in range(max_tasks)])
    ax.set_ylabel("Normalised AUC (per-task)")
    ax.set_title("Per-task AUC: higher = better plasticity on that task")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_forgetting_heatmap(summaries_by_mode: Dict[str, List[RunSummary]],
                            out_path: str) -> None:
    if not _has_mpl():
        return
    modes = [m for m, rs in summaries_by_mode.items() if rs]
    if not modes:
        return
    max_tasks = max(max(len(r.forgetting_proxy_per_task) for r in summaries_by_mode[m])
                    for m in modes)
    M = np.full((len(modes), max_tasks), np.nan)
    for i, m in enumerate(modes):
        stack = np.full((len(summaries_by_mode[m]), max_tasks), np.nan)
        for j, r in enumerate(summaries_by_mode[m]):
            stack[j, :len(r.forgetting_proxy_per_task)] = r.forgetting_proxy_per_task
        M[i] = np.nanmean(stack, axis=0)

    fig, ax = plt.subplots(figsize=(max(5, max_tasks), 0.6 * len(modes) + 1.5))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r",
                   vmin=-abs(np.nanmax(np.abs(M))),
                   vmax=abs(np.nanmax(np.abs(M))))
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels(modes)
    ax.set_xticks(range(max_tasks))
    ax.set_xticklabels([f"task {i+1}" for i in range(max_tasks)])
    for i in range(len(modes)):
        for j in range(max_tasks):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=8, color="black")
    ax.set_title("Forgetting (end-of-task - end-of-run) -- red = more forgetting")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_warmup_flag_ratio(summaries_by_mode: Dict[str, List[RunSummary]],
                           out_path: str) -> None:
    if not _has_mpl():
        return
    modes = [m for m, rs in summaries_by_mode.items() if rs]
    if not modes:
        return
    categories = ["Fast", "Meta", "Random"]
    data = np.zeros((len(modes), len(categories)))
    for i, m in enumerate(modes):
        c = Counter()
        for r in summaries_by_mode[m]:
            c.update(r.flag_history)
        total = sum(c.values()) or 1
        for j, cat in enumerate(categories):
            data[i, j] = c.get(cat, 0) / total
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bottom = np.zeros(len(modes))
    colors = {"Fast": "#1f77b4", "Meta": "#2ca02c", "Random": "#d62728"}
    for j, cat in enumerate(categories):
        ax.bar(modes, data[:, j], bottom=bottom, label=cat, color=colors[cat])
        bottom += data[:, j]
    ax.set_ylabel("Warm-up selection ratio")
    ax.set_ylim(0, 1)
    ax.set_title("Adaptive warm-up: how often each initialisation was chosen")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_hybrid_reason_pie(summaries_by_mode: Dict[str, List[RunSummary]],
                           out_path: str) -> None:
    if not _has_mpl():
        return
    runs = summaries_by_mode.get("hybrid", [])
    if not runs:
        return
    total = Counter()
    for r in runs:
        total.update(r.hybrid_reasons)
    if not total:
        return
    labels, sizes = zip(*total.most_common())
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90)
    ax.set_title("Hybrid detector: firing-reason distribution")
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_ratio_to_oracle(summaries_by_mode: Dict[str, List[RunSummary]],
                         out_path: str) -> None:
    if not _has_mpl():
        return
    if "oracle" not in summaries_by_mode or not summaries_by_mode["oracle"]:
        return
    oracle_by_seed = {r.seed: r for r in summaries_by_mode["oracle"]}
    modes = [m for m in summaries_by_mode if m != "oracle"
             and summaries_by_mode[m]]
    ratios = {}
    for m in modes:
        vals = []
        for r in summaries_by_mode[m]:
            if r.seed in oracle_by_seed:
                o = oracle_by_seed[r.seed].avg_perf_proxy
                if abs(o) > 1e-6:
                    vals.append(r.avg_perf_proxy / o)
        ratios[m] = vals
    fig, ax = plt.subplots(figsize=(5, 3.5))
    xs = list(ratios)
    mus = [np.mean(v) if v else 0.0 for v in ratios.values()]
    ses = [np.std(v, ddof=0) / math.sqrt(len(v)) if len(v) > 1 else 0.0
           for v in ratios.values()]
    ax.bar(xs, mus, yerr=ses, capsize=3)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    for i, v in enumerate(mus):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("avg_perf / oracle.avg_perf")
    ax.set_title("Boundary-free methods vs. Oracle ceiling (1.0)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_tolerance_sweep(summaries_by_mode: Dict[str, List[RunSummary]],
                         tols: Sequence[int],
                         out_path: str) -> None:
    if not _has_mpl():
        return
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for mode, runs in summaries_by_mode.items():
        if mode == "oracle" or not runs:
            continue
        f1s = []
        for t in tols:
            vals = [
                match_detections(r.detected_boundaries,
                                 r.oracle_boundaries, t)["f1"]
                for r in runs
            ]
            f1s.append(np.mean(vals))
        ax.plot(tols, f1s, marker="o", label=mode)
    ax.set_xscale("log")
    ax.set_xlabel("Detection tolerance (steps)")
    ax.set_ylabel("Mean F1")
    ax.set_title("Detection quality vs. tolerance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_pvalue_traces(summaries_by_mode: Dict[str, List[RunSummary]],
                       out_dir: str) -> None:
    if not _has_mpl():
        return
    for mode in ("swoks", "implicit"):
        runs = summaries_by_mode.get(mode, [])
        if not runs:
            continue
        # Pick the first seed with a detection log that carries pvals.
        r = next((rr for rr in runs
                  if any("pval" in e for e in rr.detection_log)), None)
        if r is None:
            continue
        steps = [e["step"] for e in r.detection_log if "pval" in e]
        logp = [-math.log(max(e["pval"], 1e-300)) for e in r.detection_log
                if "pval" in e]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.stem(steps, logp, basefmt=" ")
        for b in r.oracle_boundaries:
            ax.axvline(b, color="k", linestyle="--", alpha=0.35)
        ax.set_xlabel("Step")
        ax.set_ylabel("-log(p) at fire")
        ax.set_title(f"{mode}: log-evidence at each fire (seed={r.seed})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"pvalue_trace_{mode}.png"), dpi=160)
        plt.close(fig)


def plot_hybrid_vs_implicit_timeline(summaries_by_mode: Dict[str, List[RunSummary]],
                                     out_path: str) -> None:
    """Overlay implicit and hybrid fires to show the cascade effect."""
    if not _has_mpl():
        return
    imp_runs = summaries_by_mode.get("implicit", [])
    hyb_runs = summaries_by_mode.get("hybrid", [])
    if not imp_runs or not hyb_runs:
        return
    seeds_common = set(r.seed for r in imp_runs) & set(r.seed for r in hyb_runs)
    if not seeds_common:
        return
    seed = sorted(seeds_common)[0]
    imp = next(r for r in imp_runs if r.seed == seed)
    hyb = next(r for r in hyb_runs if r.seed == seed)
    fig, ax = plt.subplots(figsize=(9, 2.5))
    for o in imp.oracle_boundaries:
        ax.axvline(o, color="grey", linestyle="--", linewidth=1)
    for d in imp.detected_boundaries:
        ax.axvline(d, color="#d62728", alpha=0.6, linewidth=1)
    for d in hyb.detected_boundaries:
        ax.axvline(d, color="#1f77b4", alpha=0.9, linewidth=1.5)
    ax.set_yticks([])
    ax.set_xlim(0, max(imp.n_steps, hyb.n_steps))
    ax.set_xlabel("Step")
    ax.set_title(f"implicit (red) vs hybrid (blue) fires; grey dashed = oracle (seed={seed})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ======================================================================
# 7. Report generators
# ======================================================================
def write_per_seed_csv(summaries_by_mode: Dict[str, List[RunSummary]],
                       norm_forg: Dict[str, List[float]],
                       out_path: str) -> None:
    header = ("mode,seed,seq,n_steps,avg_perf_proxy,avg_perf_per_game,"
              "avg_retention,per_game_last,per_game_peak,per_game_retention,"
              "mean_auc,mean_ft,"
              "forgetting_proxy,forgetting_normalised,"
              "tp,fp,fn,precision,recall,f1,mean_delay,max_delay,"
              "num_oracle,num_detected,posthoc_avg_perf\n")
    with open(out_path, "w") as f:
        f.write(header)
        for mode, runs in summaries_by_mode.items():
            for r in runs:
                mean_auc = (float(np.mean(r.auc_per_task))
                            if r.auc_per_task else 0.0)
                mean_ft = (float(np.mean(r.ft_per_task))
                           if r.ft_per_task else 0.0)
                norm_f = np.mean(norm_forg.get(mode, []))
                posthoc = (f"{r.posthoc_avg_perf:.4f}"
                           if r.posthoc_avg_perf is not None else "")
                # Encode per-game dicts as ;-separated key=value strings so
                # CSV stays one-row-per-seed.
                pg_last = ";".join(f"{k}={v:.2f}" for k, v in r.per_game_last.items())
                pg_peak = ";".join(f"{k}={v:.2f}" for k, v in r.per_game_peak.items())
                pg_ret = ";".join(f"{k}={v:.3f}" for k, v in r.per_game_retention.items())
                d = r.detection
                f.write(
                    f"{mode},{r.seed},{r.seq},{r.n_steps},"
                    f"{r.avg_perf_proxy:.4f},{r.avg_perf_per_game:.4f},"
                    f"{r.avg_retention:.4f},{pg_last},{pg_peak},{pg_ret},"
                    f"{mean_auc:.4f},{mean_ft:.4f},"
                    f"{r.forgetting_proxy_total:.4f},"
                    f"{float(norm_f) if np.isfinite(norm_f) else 0.0:.4f},"
                    f"{d['tp']},{d['fp']},{d['fn']},"
                    f"{d['precision']:.3f},{d['recall']:.3f},{d['f1']:.3f},"
                    f"{d['mean_delay']},{d['max_delay']},"
                    f"{len(r.oracle_boundaries)},"
                    f"{len(r.detected_boundaries)},{posthoc}\n"
                )


def _agg(values, fmt="{:.3f}", *, sep: str = " +/- "):
    """Format mean/SE for a column.  Use sep='$\\pm$' for LaTeX output."""
    mu, se = mean_se(values)
    if not np.isfinite(mu):
        return "n/a"
    return f"{fmt.format(mu)}{sep}{fmt.format(se)}"


def build_paper_table(summaries_by_mode: Dict[str, List[RunSummary]],
                      norm_forg: Dict[str, List[float]],
                      *,
                      sep: str = " +/- ") -> List[Tuple[str, Dict[str, str]]]:
    """Return [(mode, {col: 'mu sep se' string})] ready for tabulation.

    Pass sep='$\\pm$' when generating the LaTeX table.
    """
    rows = []
    for mode in summaries_by_mode:
        runs = summaries_by_mode[mode]
        if not runs:
            continue
        # Avg Perf (proxy)
        ap = [r.avg_perf_proxy for r in runs]
        # Game-aware avg perf (mean across games of last-occurrence window).
        # Equally weights each game; doesn't favour whatever game is last.
        ap_pg = [r.avg_perf_per_game for r in runs]
        # Per-game retention (last_window / peak_window, averaged across games)
        # Bounded ~[0, 1]; replaces forgetting_norm as the principled forgetting
        # measure that doesn't penalise methods that learned more.
        retention = [r.avg_retention for r in runs]
        # Post-hoc Avg Perf when available
        ap_post = [r.posthoc_avg_perf for r in runs
                   if r.posthoc_avg_perf is not None]
        # Forward transfer (per-seed mean across tasks)
        ft = [float(np.mean(r.ft_per_task)) for r in runs if r.ft_per_task]
        # Forgetting normalised
        f_norm = [v for v in norm_forg.get(mode, []) if np.isfinite(v)]
        # Detection
        f1 = [r.detection["f1"] for r in runs]
        delay = [r.detection["mean_delay"] for r in runs
                 if np.isfinite(r.detection["mean_delay"])]
        tp = sum(r.detection["tp"] for r in runs)
        fp = sum(r.detection["fp"] for r in runs)
        fn = sum(r.detection["fn"] for r in runs)

        row = {
            "n_seeds": str(len(runs)),
            "avg_perf_proxy": _agg(ap, "{:.2f}", sep=sep),
            "avg_perf_per_game": _agg(ap_pg, "{:.2f}", sep=sep),
            "avg_retention": _agg(retention, "{:.3f}", sep=sep),
            "avg_perf_posthoc": _agg(ap_post, "{:.2f}", sep=sep) if ap_post else "-",
            "forward_transfer": _agg(ft, "{:.3f}", sep=sep) if ft else "-",
            "forgetting_norm": f"{float(np.mean(f_norm)):+.3f}" if f_norm else "-",
            "det_F1": _agg(f1, "{:.2f}", sep=sep) if mode != "oracle" else "-",
            "det_delay": _agg(delay, "{:.0f}", sep=sep) if mode != "oracle" else "-",
            "tp_fp_fn": f"{tp}/{fp}/{fn}" if mode != "oracle" else "-",
        }
        rows.append((mode, row))
    return rows


def write_paper_table_txt(rows, out_path: str) -> None:
    cols = ["mode", "n_seeds", "avg_perf_proxy", "avg_perf_per_game",
            "avg_retention", "avg_perf_posthoc",
            "forward_transfer", "forgetting_norm",
            "det_F1", "det_delay", "tp_fp_fn"]
    widths = {c: max(len(c), 14) for c in cols}
    for m, r in rows:
        widths["mode"] = max(widths["mode"], len(m))
        for c in cols[1:]:
            widths[c] = max(widths[c], len(r.get(c, "")))
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "-" * len(header)
    lines = [header, sep]
    for m, r in rows:
        vals = [m.ljust(widths["mode"])]
        for c in cols[1:]:
            vals.append(r.get(c, "").ljust(widths[c]))
        lines.append("  ".join(vals))
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_paper_table_tex(rows, out_path: str) -> None:
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{FAME + TSDM on MinAtar. "
        "Avg.~Perf (proxy from training trace), Avg.~Perf (post-hoc rollouts), "
        "Forward Transfer (vs.~Oracle AUC baseline), Forgetting "
        "(cross-method-std-normalised), and detection quality. "
        "Mean $\\pm$ standard error across seeds.}\n"
        "\\label{tab:fame-tsdm}\n"
        "\\begin{tabular}{lrrrrrrrr}\n"
        "\\toprule\n"
        "Method & $n$ & AvgPerf$_{\\text{proxy}}\\uparrow$ & "
        "AvgPerf$_{\\text{posthoc}}\\uparrow$ & FT$\\uparrow$ & "
        "Forgetting$\\downarrow$ & Det.\\ F1$\\uparrow$ & "
        "Det.\\ delay$\\downarrow$ & TP/FP/FN \\\\\n"
        "\\midrule\n"
    )
    body = []
    for m, r in rows:
        body.append(
            " & ".join([
                m, r["n_seeds"], r["avg_perf_proxy"],
                r["avg_perf_posthoc"], r["forward_transfer"],
                r["forgetting_norm"], r["det_F1"],
                r["det_delay"], r["tp_fp_fn"],
            ]) + " \\\\"
        )
    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    with open(out_path, "w") as f:
        f.write(header + "\n".join(body) + "\n" + footer)


def write_report_md(summaries_by_mode: Dict[str, List[RunSummary]],
                    rows,
                    plots_dir: str,
                    out_path: str,
                    *,
                    seq: int,
                    seeds: Sequence[int],
                    tolerance: int,
                    posthoc_used: bool) -> None:
    lines = []
    lines.append("# FAME + TSDM experiment report")
    lines.append("")
    lines.append(f"- Sequence: `seq={seq}`")
    lines.append(f"- Seeds: `{list(seeds)}`")
    lines.append(f"- Detection tolerance: `{tolerance}` steps")
    lines.append(f"- Post-hoc evaluation: **{'yes' if posthoc_used else 'no'}**")
    counts = {m: len(v) for m, v in summaries_by_mode.items()}
    lines.append(f"- Runs discovered: `{counts}`")
    lines.append("")
    lines.append("## Main results")
    lines.append("")
    cols = ["mode", "n_seeds", "avg_perf_proxy", "avg_perf_per_game",
            "avg_retention", "avg_perf_posthoc",
            "forward_transfer", "forgetting_norm",
            "det_F1", "det_delay", "tp_fp_fn"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for m, r in rows:
        lines.append("| " + " | ".join(
            [m] + [r.get(c, "-") for c in cols[1:]]) + " |")
    lines.append("")

    lines.append("## Visualisations")
    for fname, title in [
        ("learning_curves.png", "Learning curves (mean +/- SE)"),
        ("detection_timeline.png", "Detection timeline vs oracle"),
        ("detection_delay_hist.png", "Distribution of TP detection delays"),
        ("detection_quality.png", "Detection quality per mode"),
        ("per_task_auc.png", "Per-task AUC"),
        ("forgetting_heatmap.png", "Forgetting heatmap"),
        ("warmup_flag_ratio.png", "Adaptive warm-up selection ratio"),
        ("hybrid_reason_pie.png", "Hybrid firing-reason distribution"),
        ("ratio_to_oracle.png", "Avg-perf ratio to oracle ceiling"),
        ("tolerance_sweep.png", "F1 vs detection tolerance"),
        ("hybrid_vs_implicit_timeline.png", "Hybrid vs implicit overlay"),
        ("pvalue_trace_swoks.png", "SWOKS p-value trace"),
        ("pvalue_trace_implicit.png", "Implicit p-value trace"),
    ]:
        p = os.path.join("plots", fname)
        full = os.path.join(plots_dir, fname)
        if os.path.exists(full):
            lines.append(f"### {title}")
            lines.append(f"![{title}]({p})")
            lines.append("")
    lines.append("## Notes on metrics")
    lines.append("")
    lines.append("- **AvgPerf (proxy)** is the mean return over the last 2%"
                 " of the training trace.  KNOWN PROBLEM: this only sees the"
                 " final task in the sequence (e.g. for seq=0 only the last"
                 " 70k steps of breakout).  It rewards methods that happen to"
                 " be good at whatever the last game is, not continual-learning"
                 " ability per se.  Reported for FAME-paper comparability.")
    lines.append("- **AvgPerf (per_game)** is a fairer training-trace proxy:"
                 " for each unique game in the sequence, take the mean return"
                 " in the last 5000 steps of that game's most recent task."
                 " Then average across games.  Equally weights every game"
                 " regardless of how often or where it appears in the"
                 " sequence.  Higher = better.")
    lines.append("- **Avg Retention** is the mean across games of"
                 " (last_window / peak_window) where peak is the max"
                 " end-of-task window across all occurrences of that game."
                 " Bounded ~[0, 1].  Replaces the FAME forgetting metric"
                 " with one that doesn't penalise methods for learning more:"
                 " a method that learnt a peak of 50 and lost down to 25"
                 " has the same retention (0.5) as one that peaked at 10"
                 " and lost down to 5.")
    lines.append("- **AvgPerf (posthoc)** loads the final *meta* learner and"
                 " rolls it for N episodes in each of {breakout, space_invaders,"
                 " freeway}; averaged across games.  Matches the FAME paper's"
                 " $(1/K)\\sum_i p_i(K\\cdot T)$ exactly.  This is the gold"
                 " standard when MetaFinal.pt weights are available.")
    lines.append("- **Forward Transfer** compares per-task AUC against the"
                 " baseline mode (Oracle by default, standing in for Reset).")
    lines.append("- **Forgetting (norm)** is the FAME paper's metric:"
                 " per-task (end-of-task - end-of-run), normalised by the"
                 " cross-method std per task.  KNOWN PROBLEM: the end-of-run"
                 " window is on whatever game is *last*, while end-of-task is"
                 " on a potentially different game.  Reported for paper"
                 " comparability; prefer Avg Retention.")
    lines.append("- **Detection F1** uses greedy 1-to-1 matching of detected"
                 " boundaries to oracle switches within `tolerance` steps.")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ======================================================================
# 8. CLI entry-point
# ======================================================================
def run_experiment(args: argparse.Namespace) -> None:
    seeds = list(args.seeds)
    modes = [m for m in DETECTOR_MODES if m in args.modes]

    paths = discover_runs(args.results_dir, args.seq, seeds, modes)
    print("Discovered runs:")
    for m in modes:
        print(f"  {m:<9s}: {sorted(paths[m])}")
    if not any(paths.values()):
        print(f"[experiment] No FAME_*_seq_{args.seq}_*_returns.pkl "
              f"found in {args.results_dir}/ for seeds {seeds}")
        return

    # Load + summarise.
    summaries_by_mode: Dict[str, List[RunSummary]] = {m: [] for m in modes}
    for m in modes:
        for s, p in sorted(paths[m].items()):
            rs = summarise_run(p, tolerance=args.tolerance)
            summaries_by_mode[m].append(rs)

    # Populate FT using the chosen baseline mode.
    compute_forward_transfer(summaries_by_mode, baseline=args.baseline_mode)
    norm_forg = normalise_forgetting_across_methods(summaries_by_mode)

    # Optional: post-hoc evaluation.
    if args.eval_posthoc:
        if not _HAS_TORCH:
            print("[experiment] --eval-posthoc requested but torch is not "
                  "available; skipping.")
        else:
            print("[experiment] Running post-hoc rollouts "
                  f"({args.posthoc_episodes} episodes per game)...")
            for m, runs in summaries_by_mode.items():
                for r in runs:
                    try:
                        per_game = posthoc_evaluate(
                            args.models_dir, r.filename_stem,
                            games=GAMES,
                            n_episodes=args.posthoc_episodes,
                            max_steps_per_ep=args.posthoc_max_steps,
                            device=args.device,
                            in_channels=7,
                            num_actions=6,
                            eval_seed=args.posthoc_eval_seed,
                        )
                    except FileNotFoundError as e:
                        print(f"  [{m}/seed={r.seed}] skip: {e}")
                        continue
                    except Exception as e:
                        print(f"  [{m}/seed={r.seed}] eval error: {e}")
                        continue
                    r.posthoc_per_game = per_game
                    r.posthoc_avg_perf = float(np.mean(
                        [mu for (mu, _) in per_game.values()]))
                    print(f"  [{m}/seed={r.seed}] avg={r.posthoc_avg_perf:.2f} "
                          f"per_game={ {k: round(v[0],2) for k,v in per_game.items()} }")

    # Create output directory.
    tag = args.tag or f"seq{args.seq}_seeds{'_'.join(str(s) for s in seeds)}"
    out_dir = args.out_dir or os.path.join(args.results_dir,
                                           f"experiment_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = _ensure_plots_dir(out_dir)

    # CSV + tables.
    csv_path = os.path.join(out_dir, "metrics_per_seed.csv")
    write_per_seed_csv(summaries_by_mode, norm_forg, csv_path)
    print(f"[experiment] CSV -> {csv_path}")

    rows = build_paper_table(summaries_by_mode, norm_forg, sep=" +/- ")
    rows_tex = build_paper_table(summaries_by_mode, norm_forg,
                                 sep=" $\\pm$ ")
    write_paper_table_txt(rows, os.path.join(out_dir, "paper_table.txt"))
    write_paper_table_tex(rows_tex, os.path.join(out_dir, "paper_table.tex"))
    print(f"[experiment] LaTeX -> {os.path.join(out_dir, 'paper_table.tex')}")

    # Aggregate CSV (for quick reference).
    agg_path = os.path.join(out_dir, "metrics_aggregate.csv")
    with open(agg_path, "w") as f:
        f.write("mode,n,avg_perf_proxy,avg_perf_per_game,avg_retention,"
                "avg_perf_posthoc,forward_transfer,"
                "forgetting_norm,det_F1,det_delay,TP,FP,FN\n")
        for m, r in rows:
            f.write(",".join([m, r["n_seeds"], r["avg_perf_proxy"],
                              r["avg_perf_per_game"], r["avg_retention"],
                              r["avg_perf_posthoc"], r["forward_transfer"],
                              r["forgetting_norm"], r["det_F1"],
                              r["det_delay"], r["tp_fp_fn"]]) + "\n")

    # Plot gallery.
    if args.plots:
        print("[experiment] Rendering plots -> ", plots_dir)
        plot_learning_curves(summaries_by_mode,
                             os.path.join(plots_dir, "learning_curves.png"),
                             smooth_window=args.smooth)
        plot_detection_timeline(summaries_by_mode,
                                os.path.join(plots_dir, "detection_timeline.png"))
        plot_detection_delay_hist(summaries_by_mode,
                                  os.path.join(plots_dir, "detection_delay_hist.png"))
        plot_detection_quality(summaries_by_mode,
                               os.path.join(plots_dir, "detection_quality.png"))
        plot_per_task_auc(summaries_by_mode,
                          os.path.join(plots_dir, "per_task_auc.png"))
        plot_forgetting_heatmap(summaries_by_mode,
                                os.path.join(plots_dir, "forgetting_heatmap.png"))
        plot_warmup_flag_ratio(summaries_by_mode,
                               os.path.join(plots_dir, "warmup_flag_ratio.png"))
        plot_hybrid_reason_pie(summaries_by_mode,
                               os.path.join(plots_dir, "hybrid_reason_pie.png"))
        plot_ratio_to_oracle(summaries_by_mode,
                             os.path.join(plots_dir, "ratio_to_oracle.png"))
        plot_tolerance_sweep(summaries_by_mode, args.tolerance_sweep,
                             os.path.join(plots_dir, "tolerance_sweep.png"))
        plot_hybrid_vs_implicit_timeline(
            summaries_by_mode,
            os.path.join(plots_dir, "hybrid_vs_implicit_timeline.png"))
        plot_pvalue_traces(summaries_by_mode, plots_dir)

    # Markdown report.
    report_path = os.path.join(out_dir, "report.md")
    write_report_md(summaries_by_mode, rows, plots_dir, report_path,
                    seq=args.seq, seeds=seeds, tolerance=args.tolerance,
                    posthoc_used=args.eval_posthoc)
    print(f"[experiment] Report -> {report_path}")

    # Pickle everything for downstream reuse.
    blob_path = os.path.join(out_dir, "summaries.pkl")
    with open(blob_path, "wb") as f:
        pickle.dump({
            "summaries": summaries_by_mode,
            "norm_forg": norm_forg,
            "paper_table": rows,
            "args": vars(args),
        }, f)
    print(f"[experiment] Pickle -> {blob_path}")

    # Stdout summary.
    print("\n=== Summary ===")
    print(open(os.path.join(out_dir, "paper_table.txt")).read())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Unified FAME + TSDM experiments / results generator"
    )
    ap.add_argument("--results_dir", default="results",
                    help="Directory containing FAME_*_returns.pkl files")
    ap.add_argument("--models_dir", default="models",
                    help="Directory with {filename}_MetaFinal.pt weights "
                         "(only used when --eval-posthoc is set)")
    ap.add_argument("--seq", type=int, default=0,
                    help="Sequence id to aggregate")
    ap.add_argument("--seeds", type=int, nargs="+", required=True,
                    help="List of seeds to include")
    ap.add_argument("--modes", nargs="+", default=list(DETECTOR_MODES),
                    help="Subset of detector modes to include")
    ap.add_argument("--tolerance", type=int, default=60000,
                    help="Detection tolerance (steps) for TP")
    ap.add_argument("--tolerance_sweep", type=int, nargs="+",
                    default=[1000, 5000, 15000, 60000, 120000, 240000],
                    help="Tolerances for the F1 vs tolerance plot")
    ap.add_argument("--baseline_mode", default=BASELINE_MODE,
                    help="Mode to use as the AUC baseline for Forward Transfer")
    ap.add_argument("--smooth", type=int, default=3000,
                    help="Smoothing window for learning-curve plot")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory (default: results/experiment_<tag>)")
    ap.add_argument("--tag", default=None,
                    help="Short tag appended to out_dir")
    ap.add_argument("--plots", dest="plots", action="store_true", default=True,
                    help="Render plots (default)")
    ap.add_argument("--no-plots", dest="plots", action="store_false",
                    help="Skip plot rendering")

    # Post-hoc evaluation
    ap.add_argument("--eval-posthoc", action="store_true",
                    help="Roll out the saved final meta policy in each game")
    ap.add_argument("--posthoc_episodes", type=int, default=20,
                    help="Episodes per game for post-hoc evaluation")
    ap.add_argument("--posthoc_max_steps", type=int, default=2000,
                    help="Max steps per episode during post-hoc eval")
    ap.add_argument("--posthoc_eval_seed", type=int, default=9999,
                    help="Seed for post-hoc env rollouts")
    ap.add_argument("--device", default="cpu",
                    help="torch device for post-hoc eval (cpu/cuda)")

    return ap.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
