# Experimental design

The v2 protocol defines the three-domain benchmark. The v3 addendum freezes the external target, primary contrast and temporal sensitivity analysis. Earlier outcomes on the three benchmark domains and the UCI dataset description were already known. The freeze is a local timestamp and hash record. [v2](../protocol/v2_protocol.json) · [v3](../protocol/v3_protocol.json) · [Freeze record](../protocol/freeze_record.json)

## Controlled comparison

Hashing the full harmonized feature vector modulo ten assigns groups 0–5 to training, 6–7 to calibration and 8–9 to testing. Large groups of identical values can make the realized row proportions uneven. The same target rows serve every source condition and seed. Both feature variants use this partition, including when protocol-only vectors coincide across partitions. [Partition composition](../supplement/partition_composition.csv) · [Canonical checks](../supplement/canonical_split_checks.csv)

Each target has four source conditions. The budget is the smaller of 60,000 and the smallest available domain training count, yielding 60,000 in every controlled condition, including pooled training. Learned models give each source/class pair equal total weight. Majority uses the unweighted training frequencies. Median imputation and scaling use training data. Seeds 42, 7 and 2026 give 360 fits across five algorithms and two representations. Two thresholds per fit yield 720 rows.

`minimal_core` adds log1p duration to TCP, UDP, ICMP and other indicators. UNSW uses `dur`, ToN uses `duration`, CIC uses `flow_duration` with protocol indicators thresholded above 0.5. CIC's separately named `Duration` field represents TTL. `protocol_only` removes duration. The paired differences measure sensitivity to this feature under the supplied exporters. [Duration profiles](../supplement/duration_profiles.csv) · [Paired results](../tables/duration_paired.csv)

## Operating points and calendar shifts

At the fixed operating point, a score of at least 0.5 triggers an alarm. For a source benign-tail budget α, sort the n benign scores, take the one-based order statistic at ceil((1−α)n), and use the next representable number above it. Pooling takes the maximum source threshold. The standard budget is 1%. Test labels never determine that cutoff.

The UTC train, calibration and test days appear in the [split audit](../run_results/strict_temporal_split_audit.csv). Groups recurring across periods belong to the earliest period. The separate native-feature analysis compares full and conservative predictors. Calibration sensitivity holds the fitted logistic regression and LightGBM models and their test sets fixed, varying the full calibration day or either half and budgets of 1%, 0.5% and 0.1%. A window requires at least 30 benign observations. [Feature manifest](../run_results/native_feature_manifest.json) · [Window counts and outcomes](../run_results/calibration_sensitivity.csv)

## External target and interpretation

The v3 study scores every RT-IoT2022 record in original order with the controlled models fitted from UNSW, ToN, or both while holding out CIC. It uses the frozen protocol mapping, log duration and source thresholds. Ninety sets of scores give 180 metric rows. The primary contrast compares pooled LightGBM against **each** single source on mean Macro-F1 at 0.5 with duration. [External contrasts](../tables/external_pooling.csv)

Macro-F1 weights the two class-specific F1 scores equally. Recall divides detected attacks by all attacks. FPR divides benign alarms by all benign records. The report also records ROC-AUC, average precision and F1 standardized to 20% attack prevalence. The 28 historical paired comparisons and clustered intervals describe conditional variation on the available test data. Neither record-level counts nor repeated seeds establish independent testbeds. [Statistical comparisons](../run_results/paired_statistical_comparisons.csv) · [Conditional intervals](../run_results/conditional_cluster_confidence_intervals.csv)
