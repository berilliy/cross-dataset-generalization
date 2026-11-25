# Obtaining the experimental inputs

The Git checkout includes schemas, sample identities by file hash and aggregate outputs. Obtain the three exact samples from the research package where authorized. Public parent collections and the UCI download are listed in the [dataset register](../docs/DATASETS.md). No public download for the exact three cleaned samples was established.

## Training and external evaluation

Restore these paths relative to the notebook. The corresponding manifests and schema files already reside beside their expected data directories.

```text
training_samples/
  sample_export.json
  sample_sha256.csv
  upstream_parquet_sha256.csv
  UNSW_NB15/data/sample.parquet
  TON_IOT/data/sample.parquet
  CICIOT2023/data/sample.parquet
external_data/rt-iot2022.zip
```

Keep the UCI ZIP intact, including its CSV member named `RT_IOT2022`. The following PowerShell commands fetch the archive and compare its SHA-256 with the frozen release record. Run from the checkout root.

```powershell
$release = Get-Content -Raw external_data/release.json | ConvertFrom-Json
Invoke-WebRequest $release.url -OutFile external_data/rt-iot2022.zip
if ((Get-FileHash external_data/rt-iot2022.zip).Hash -ne $release.sha256) { throw "UCI release hash mismatch" }
```

Alternatively, point `NIDS_CLEAN_ROOT` at a complete sample root with the same subdirectories. Keep `sample_export.json` present so the notebook verifies the [sample hashes](../training_samples/sample_sha256.csv) and consumes each sample once.

## Rebuilding samples from the cleaned mirrors

The historical raw-cleaning notebook and full cleaned shards are absent. To start from the cleaned stage, obtain the exact original `nids_clean_v1` directory, with per-dataset `manifest.json`, `schema.json` and `data/*.parquet`. Compare all shard sizes and hashes with the [upstream manifest](../training_samples/upstream_parquet_sha256.csv). Use the recorded export environment (Python 3.13.15, pandas 2.3.3 and NumPy 2.1.3). In Kaggle, attach that one complete root or place it at `/kaggle/working/nids_clean_v1`, then execute [export_kaggle_samples.py](../source/export_kaggle_samples.py) as a notebook cell. Extract its output ZIP into the sample root.

The exporter hashes the source filename and row number before the experimental splits. It takes the low 32 bits divided by 2³² and keeps values below 0.07 for UNSW, 0.015 for ToN and 0.025 for CIC. Selection is label-independent. Use the sample manifest to check exact file identity. A matching row count alone cannot establish equivalence.

## Historical replay and numerical verification

For replay, restore `saved_results/strict_temporal_predictions.parquet` and `saved_results/unseen_predictions.parquet` from the delivered package alongside the two same-stem metrics CSVs. `NIDS_SAVED` can point to this complete directory. For score-level verification use the original package's `run_results` tree, which also contains split memberships, per-fit scores and calibration scores. Historical and corrected predictions have different hashes. [Full package manifest](../package_sha256.csv) · [Execution instructions](../docs/REPRODUCIBILITY.md)
