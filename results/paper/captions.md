# Figure and table captions

## Main figures

**Figure 1** (`fig1_data_splits`). Dataset and benchmark splits (full profile: 20,926 records, 575 events, 4,026 stations). (a) Magnitude versus rupture distance by hypocentre region (record counts in the legend). (b) Station $V_{S30}$; dotted lines mark the NEHRP site-class boundaries E|D at 180 m/s, D|C at 360 m/s, C|B at 760 m/s and B|A at 1500 m/s (only boundaries within the data range are drawn). (c) Usable records per period after masking $T > 1/\mathrm{LUF}$. (d) Share of test records whose event or station also occurs in training, averaged over the folds of each split scheme. Split schemes: Rand = record-level random split (leaky reference); Evt = event-grouped 5-fold; Stn = station-grouped 5-fold; Reg = leave-one-region-out; Temp = temporal (train before 2008, test 2008 onward); $M_w{\geq}7$ = magnitude hold-out (test events with $M_w \geq 7$); CL1$\rightarrow$2 = mainshocks (CL1) for training, aftershocks (CL2) for testing.

**Figure 2** (`fig2_variance_components`). Period-dependent variance components of test residuals: (a) between-event $\tau(T)$, (b) site-to-site $\phi_{S2S}(T)$ and (c) single-station within-event $\phi_0(T)$, estimated by crossed random effects on held-out events (Event split, full profile), averaged over folds.

**Figure 3** (`fig3_regime_accuracy`). Accuracy across distribution-shift regimes (full profile): (a) test RMSE and (b) empirical coverage of nominal 90% Gaussian intervals (dashed line), averaged over folds and 27 intensity measures, for classical NGA-West2 models (train_fit bias term), the mixed-effects baseline and machine-learning models. Hollow markers (site) are the NGA-West2 models with train_fit station terms added (zero for stations unseen in train_fit) and the matching single-station sigma. Split schemes: Evt = event-grouped 5-fold; Stn = station-grouped 5-fold; Reg = leave-one-region-out; Temp = temporal (train before 2008, test 2008 onward); $M_w{\geq}7$ = magnitude hold-out (test events with $M_w \geq 7$); CL1$\rightarrow$2 = mainshocks (CL1) for training, aftershocks (CL2) for testing.

**Figure 4** (`fig4_conformal_ablation`). Uncertainty-method ablation at nominal 90% (full profile): (a) empirical coverage (dashed line: nominal) and (b) mean interval width, averaged over folds and 27 intensity measures. Markers are the mean over 8 models; bars span the model range. The dashed line in (a) is the nominal 90% target. Split schemes: Evt = event-grouped 5-fold; Stn = station-grouped 5-fold; Reg = leave-one-region-out; Temp = temporal (train before 2008, test 2008 onward); $M_w{\geq}7$ = magnitude hold-out (test events with $M_w \geq 7$); CL1$\rightarrow$2 = mainshocks (CL1) for training, aftershocks (CL2) for testing.

**Figure 5** (`fig5_spectral_bands`). Example test response spectra (RotD50) with nominal 90% bands from Hybrid ASK14+MLP (full profile; rows: Event split, fold 2; Region split, held-out region California WUS). Shaded: HSCP marginal band; dotted: split conformal band; dash-dot: HSCP simultaneous band over all usable periods; dashed: model median (HSCP re-centres records at stations with a calibration-set site term). Records are chosen at the 10th, 50th and 90th magnitude percentiles among test records with at least 90% usable periods.

**Figure 6** (`fig6_conditional_coverage`). Conditional coverage at nominal 90% (dashed) by (a) magnitude, (b) rupture distance (km), (c) NEHRP $V_{S30}$ class, ordered from stiff rock to soft soil (A/B: $\geq$760 m/s; C: 360–760 m/s; D: 180–360 m/s; E: <180 m/s), and (d) whether the test station was seen in training or calibration (full profile), pooled over split schemes, models, folds and intensity measures.

## Supplementary figures

**Figure S1** (`figS1_random_leakage_inflation`). Leakage inflation: test RMSE under a record-level random split versus the event-grouped split (full profile), averaged over folds and IMs. Labels give the RMSE increase when events are held out, i.e. the optimism of random-split evaluation.

**Figure S2** (`figS2_period_coverage_heatmap`). Coverage at nominal 90% per intensity measure and split scheme (full profile), averaged over models and folds: (a) split conformal, (b) HSCP. The colour scale is monotonic and colour-blind safe (cividis); values are printed in each cell. Split schemes: Evt = event-grouped 5-fold; Stn = station-grouped 5-fold; Reg = leave-one-region-out; Temp = temporal (train before 2008, test 2008 onward); $M_w{\geq}7$ = magnitude hold-out (test events with $M_w \geq 7$); CL1$\rightarrow$2 = mainshocks (CL1) for training, aftershocks (CL2) for testing.

## Tables

**table1_attrition**. Record selection from the NGA-West2 flatfiles, rule by rule (full profile, then the sp23-specific steps).

**table2_splits**. Benchmark split schemes (full profile): mean records per fold and the share of test records whose event or station also occurs in training.

**table3_accuracy**. Test RMSE (ln units) by model and split scheme (full profile), averaged over folds and 27 intensity measures.

**table4_conformal_ablation**. Empirical coverage and mean interval width (ln units, in parentheses) by uncertainty method and split scheme (full profile), averaged over 8 models, folds and 27 intensity measures.

**table5_conditional_coverage**. Conditional coverage at nominal 90% by magnitude, rupture distance, VS30 class and station status (full profile), pooled over schemes, models and folds.

**tableS1_per_im_metrics**. Per-IM test RMSE (ln units; one column per model) and calibrated-band coverage and width at 90% (Event split, full profile).

**tableS2_sp23_comparison**. Profile comparison: full versus sp23 (the Sedaghati and Pezeshk 2023 selection) and test RMSE under the Event split.

**tableS3_bootstrap_cis**. Event-clustered bootstrap 95% confidence intervals (full profile; 1,000 replicates resampling whole earthquakes within each split scheme): estimate [lower, upper].

**tableS4_station_rmse**. Test RMSE (ln units) at stations seen and unseen in train_fit (full profile), for models that use station information: estimate [event-clustered bootstrap 95% interval].
