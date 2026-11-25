# Repository and evidence guide

The repository preserves the paths that the notebook, scripts, report and historical manifests resolve from the project root. The multilingual source materials retain their original filenames.

| Path | Contents |
|---|---|
| [Notebook](../notebook3ec3218ec0_corrected.ipynb), [source](../source) | Executed notebook, runner, source modules, report builder and verifiers |
| [Protocol](../protocol) | v2 benchmark, v3 addendum, local freeze records and post-evaluation amendment |
| [Report](../research.html), [editable report](../research_corrected.txt) | Research report with five embedded figures and its editable text |
| [Tables](../tables), [figures](../figures) | Paper tables and five PNG/PDF figure pairs |
| [Run results](../run_results) | Corrected run metrics, evidence of execution and eight diagnostic plots |
| [Supplement](../supplement) | Methods, checks, bibliography, first-v3 comparison tables and exact repeat-run record |
| [Saved results](../saved_results) | Historical metrics, cleaning manifests and replay reference metadata |
| [Training metadata](../training_samples), [UCI release record](../external_data/release.json) | Input schemas and checksums |
| [Data](../data), [docs](.), [tests](../tests) | Acquisition guide, publication documentation and read-only validation |
| [Original README](../README.txt), [changelog](../CHANGELOG.txt) | Historical instructions for the complete private package |
| [Package manifest](../package_sha256.csv), [publication manifest](release_sha256.csv) | Byte identities of the original delivery and research files included here |

## Evidence by question

| Analysis | Files |
|---|---|
| Within-domain vs transfer | [Controlled benchmark](../run_results/controlled_benchmark.csv), [performance drop](../run_results/benchmark_performance_drop.csv) |
| Model vs Majority and duration ablation | [Paired Majority differences](../tables/majority_paired.csv), [paired duration differences](../tables/duration_paired.csv) |
| Feature shift | [Nine transfer conditions](../tables/shift_conditions.csv), [class-conditional shift](../run_results/benchmark_class_conditional_shift.csv) |
| Calendar holdout and native ablation | [Temporal metrics](../run_results/strict_temporal_metrics.csv), [native ablation](../run_results/native_ablation.csv), [calibration sensitivity](../run_results/calibration_sensitivity.csv) |
| Single vs pooled | [Three-domain pooling](../tables/benchmark_pooling.csv), [external summary](../tables/external_summary.csv), [external pooling](../tables/external_pooling.csv) |
| Statistical and numerical checks | [Paired comparisons](../run_results/paired_statistical_comparisons.csv), [export checks](../run_results/independent_export_verification.csv), [external and calibration checks](../supplement/verification_new.csv), [repeat comparison](../supplement/repeat_run_comparison.csv) |
