# FAME + TSDM experiment report

- Sequence: `seq=0`
- Seeds: `[1]`
- Detection tolerance: `60000` steps
- Post-hoc evaluation: **no**
- Runs discovered: `{'oracle': 1, 'swoks': 1, 'implicit': 1, 'hybrid': 1}`

## Main results

| mode | n_seeds | avg_perf_proxy | avg_perf_posthoc | forward_transfer | forgetting_norm | det_F1 | det_delay | tp_fp_fn |
|---|---|---|---|---|---|---|---|---|
| oracle | 1 | 11.94 +/- 0.00 | - | 0.000 +/- 0.000 | +1.260 | - | - | - |
| swoks | 1 | 9.17 +/- 0.00 | - | -0.865 +/- 0.000 | +2.136 | 0.91 +/- 0.00 | 4983 +/- 0 | 5/0/1 |
| implicit | 1 | 5.02 +/- 0.00 | - | -0.997 +/- 0.000 | +2.814 | 0.12 +/- 0.00 | 20939 +/- 0 | 4/56/2 |
| hybrid | 1 | 5.50 +/- 0.00 | - | -1.148 +/- 0.000 | +2.787 | 0.22 +/- 0.00 | 13010 +/- 0 | 4/26/2 |

## Visualisations
### Learning curves (mean +/- SE)
![Learning curves (mean +/- SE)](plots/learning_curves.png)

### Detection timeline vs oracle
![Detection timeline vs oracle](plots/detection_timeline.png)

### Distribution of TP detection delays
![Distribution of TP detection delays](plots/detection_delay_hist.png)

### Detection quality per mode
![Detection quality per mode](plots/detection_quality.png)

### Per-task AUC
![Per-task AUC](plots/per_task_auc.png)

### Forgetting heatmap
![Forgetting heatmap](plots/forgetting_heatmap.png)

### Adaptive warm-up selection ratio
![Adaptive warm-up selection ratio](plots/warmup_flag_ratio.png)

### Hybrid firing-reason distribution
![Hybrid firing-reason distribution](plots/hybrid_reason_pie.png)

### Avg-perf ratio to oracle ceiling
![Avg-perf ratio to oracle ceiling](plots/ratio_to_oracle.png)

### F1 vs detection tolerance
![F1 vs detection tolerance](plots/tolerance_sweep.png)

### Hybrid vs implicit overlay
![Hybrid vs implicit overlay](plots/hybrid_vs_implicit_timeline.png)

### SWOKS p-value trace
![SWOKS p-value trace](plots/pvalue_trace_swoks.png)

### Implicit p-value trace
![Implicit p-value trace](plots/pvalue_trace_implicit.png)

## Notes on metrics

- **AvgPerf (proxy)** is the mean return over the last 2% of the training trace.  Fast to compute, noisy on short runs.
- **AvgPerf (posthoc)** loads the final *meta* learner and rolls it for N episodes in each of {breakout, space_invaders, freeway}; averaged across games.  Matches the FAME paper's $(1/K)\sum_i p_i(K\cdot T)$ exactly.
- **Forward Transfer** compares per-task AUC against the baseline mode (Oracle by default, standing in for Reset).
- **Forgetting** is the per-task end-of-task minus end-of-run training-trace mean, then normalised by the cross-method standard deviation per task (as in the paper).
- **Detection F1** uses greedy 1-to-1 matching of detected boundaries to oracle switches within `tolerance` steps.
