

import argparse
import glob
import os
import pickle
from collections import defaultdict

import numpy as np

from swoks_detector import match_detections


DETECTOR_MODES = ["oracle", "swoks", "implicit", "hybrid"]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def normalized_returns(returns, lo=None, hi=None):
    """Min-max normalise a return trace to [0, 1] for AUC-style comparison."""
    if lo is None:
        lo = float(np.min(returns))
    if hi is None:
        hi = float(np.max(returns))
    if hi - lo < 1e-9:
        return np.zeros_like(returns, dtype=np.float64)
    return (returns.astype(np.float64) - lo) / (hi - lo)


def auc_per_task(returns, boundaries, total_steps):
    """AUC of (normalised) returns inside each task window."""
    edges = [0] + list(boundaries) + [total_steps]
    auc = []
    for s, e in zip(edges[:-1], edges[1:]):
        if e <= s:
            continue
        auc.append(float(returns[s:e].mean()))
    return auc


def forward_transfer(curr_auc, base_auc):
    """Per-task FT = (AUC_i - AUC_b_i) / (1 - AUC_b_i)."""
    ft = []
    for c, b in zip(curr_auc, base_auc):
        denom = 1.0 - b
        ft.append((c - b) / denom if abs(denom) > 1e-6 else 0.0)
    return ft


def forgetting(returns, boundaries, total_steps, eval_window=5000):
    """Very rough forgetting proxy: performance gap between end-of-task and
    end-of-training, averaged over tasks.  Uses a short window at each."""
    edges = list(boundaries) + [total_steps]
    fs = []
    end_win = returns[max(0, total_steps - eval_window):total_steps].mean()
    for e in edges[:-1]:
        e = int(e)
        win = returns[max(0, e - eval_window):e].mean()
        fs.append(win - end_win)
    return float(np.mean(fs)) if fs else 0.0


def summarise(run, *, tolerance):
    r = np.asarray(run["returns"])
    oracle = sorted(run["oracle_boundaries"])
    detected = sorted(run["detected_boundaries"])
    total = len(r)

    detect = match_detections(detected, oracle, tolerance=tolerance)
    auc = auc_per_task(normalized_returns(r), oracle, total)
    avg_perf = float(r[-max(1, total // 50):].mean())
    forget = forgetting(r, oracle, total)

    return {
        "mode": run.get("mode", "?"),
        "avg_perf": avg_perf,
        "auc": auc,
        "mean_auc": float(np.mean(auc)) if auc else 0.0,
        "forgetting": forget,
        "oracle": oracle,
        "detected": detected,
        "detection": detect,
        "flag_history": run.get("flag_history", []),
    }


def discover(results_dir, mode, seq, seeds):
    pattern = os.path.join(
        results_dir, f"FAME_{mode}_*seq_{seq}*seed_*_returns.pkl"
    )
    out = {}
    for p in glob.glob(pattern):
        for s in seeds:
            if f"seed_{s}_" in p or p.endswith(f"seed_{s}_returns.pkl"):
                out[s] = p
    return out


def _print_mode_runs(mode, summ):
    if not summ:
        return
    print(f"\n=== {mode} runs ===")
    for s, rs in sorted(summ.items()):
        base = (f"  seed={s} avg_perf={rs['avg_perf']:.2f} "
                f"mean_auc={rs['mean_auc']:.3f} "
                f"forget={rs['forgetting']:.3f}")
        if mode != "oracle":
            d = rs["detection"]
            base += (f" TP={d['tp']} FP={d['fp']} FN={d['fn']} "
                     f"F1={d['f1']:.2f} mean_delay={d['mean_delay']}")
        print(base)


def _agg_avg_perf(summ):
    return [s["avg_perf"] for s in summ.values()]


def _agg_det(summ, key):
    return [s["detection"][key] for s in summ.values()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--seq", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--tolerance", type=int, default=60000,
                    help="Max delay (steps) for a detection to count as TP")
    args = ap.parse_args()

    # Discover runs for every detector mode we know about.
    runs_by_mode = {}
    for mode in DETECTOR_MODES:
        runs_by_mode[mode] = discover(args.results_dir, mode, args.seq,
                                      args.seeds)

    if not any(runs_by_mode.values()):
        print(f"No results found in {args.results_dir}/")
        return

    # Summarise each run.
    summ_by_mode = {}
    for mode in DETECTOR_MODES:
        summ_by_mode[mode] = {}
        for s, p in runs_by_mode[mode].items():
            summ_by_mode[mode][s] = summarise(load_pickle(p),
                                              tolerance=args.tolerance)
        _print_mode_runs(mode, summ_by_mode[mode])

    # Aggregate per-mode.
    print("\n=== Aggregate (mean +/- std over seeds) ===")
    for mode in DETECTOR_MODES:
        ss = summ_by_mode[mode]
        if not ss:
            continue
        ap_vals = _agg_avg_perf(ss)
        line = (f"{mode:<9s}: avg_perf={np.mean(ap_vals):.2f} "
                f"+/- {np.std(ap_vals):.2f}  "
                f"mean_auc={np.mean([s['mean_auc'] for s in ss.values()]):.3f}"
                f"  forget={np.mean([s['forgetting'] for s in ss.values()]):.3f}")
        if mode != "oracle":
            line += (f"  | TP={np.sum(_agg_det(ss,'tp'))} "
                     f"FP={np.sum(_agg_det(ss,'fp'))} "
                     f"FN={np.sum(_agg_det(ss,'fn'))} "
                     f"F1={np.mean(_agg_det(ss,'f1')):.3f} "
                     f"mean_delay={np.nanmean([d['mean_delay'] for d in [s['detection'] for s in ss.values()]]):.1f}")
        print(line)

    # Relative performance ratio vs. oracle ceiling.
    oracle = summ_by_mode.get("oracle", {})
    if oracle:
        print("\n=== Ratio to Oracle (1.0 = ceiling) ===")
        for mode in ("swoks", "implicit", "hybrid"):
            ss = summ_by_mode[mode]
            common = sorted(set(oracle) & set(ss))
            if not common:
                continue
            rel = []
            for s in common:
                o = oracle[s]["avg_perf"]
                k = ss[s]["avg_perf"]
                rel.append(k / o if abs(o) > 1e-6 else 0.0)
            print(f"  {mode:<9s}: {np.mean(rel):.3f}  "
                  f"over {len(common)} matched seeds")

    # CSV dump.
    csv_path = os.path.join(args.results_dir,
                            "compare_oracle_vs_swoks.csv")
    with open(csv_path, "w") as f:
        f.write("mode,seed,avg_perf,mean_auc,forgetting,"
                "tp,fp,fn,f1,mean_delay\n")
        for mode in DETECTOR_MODES:
            for s, rs in sorted(summ_by_mode[mode].items()):
                if mode == "oracle":
                    f.write(f"{mode},{s},{rs['avg_perf']:.4f},"
                            f"{rs['mean_auc']:.4f},"
                            f"{rs['forgetting']:.4f},,,,,\n")
                else:
                    d = rs["detection"]
                    f.write(f"{mode},{s},{rs['avg_perf']:.4f},"
                            f"{rs['mean_auc']:.4f},"
                            f"{rs['forgetting']:.4f},{d['tp']},{d['fp']},"
                            f"{d['fn']},{d['f1']:.3f},{d['mean_delay']}\n")
    print(f"\nCSV -> {csv_path}")


if __name__ == "__main__":
    main()
