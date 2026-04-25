# FAME + TSDM experiment report

- Sequence: `seq=0`
- Seeds: `[1, 2]`
- Detection tolerance: `2000` steps
- Post-hoc evaluation: **yes**
- Runs discovered: `{'oracle': 2, 'swoks': 2, 'implicit': 2, 'hybrid': 2}`

## Main results

| mode | n_seeds | avg_perf_proxy | avg_perf_posthoc | forward_transfer | forgetting_norm | det_F1 | det_delay | tp_fp_fn |
|---|---|---|---|---|---|---|---|---|
| oracle | 2 | 4.11 +/- 0.65 | 2.08 +/- 0.05 | -0.000 +/- 0.035 | -2.680 | - | - | - |
| swoks | 2 | 2.16 +/- 1.73 | 1.15 +/- 0.42 | -0.023 +/- 0.044 | -2.135 | 0.83 +/- 0.17 | 666 +/- 50 | 6/3/0 |
| implicit | 2 | 0.57 +/- 0.06 | 2.23 +/- 0.53 | -0.092 +/- 0.081 | -0.871 | 0.24 +/- 0.07 | 1174 +/- 125 | 3/17/2 |
| hybrid | 2 | 2.77 +/- 0.92 | 2.22 +/- 0.65 | -0.139 +/- 0.106 | -1.641 | 0.25 +/- 0.25 | 350 +/- 0 | 2/7/3 |

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
