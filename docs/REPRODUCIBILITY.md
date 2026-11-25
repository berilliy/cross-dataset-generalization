# Reproducing and checking the results

| Task | Inputs | Evidence |
|---|---|---|
| Inspect and check published tables | Git checkout and Python 3.11+ | Read-only tests compare counts, metrics, protocol identity and publication hashes |
| Replay historical comparisons | Two saved prediction Parquets and their metrics CSVs | All 14 notebook code cells ran locally in replay mode during packaging |
| Recalculate metrics from corrected scores | Complete corrected result exports, original samples and UCI ZIP | Original arithmetic and split validators passed locally during packaging |
| Train the controlled models again | Exact sampled inputs, UCI ZIP and pinned environment | The original project records two corrected, same-machine training runs. Packaging performed no training |
| Replicate from raw captures or in another network | Historical cleaning workflow, corresponding captures or independent collection | Outside the demonstrated evidence |

## Environment and execution

Run the following from the root of a **disposable working copy** with Python 3.14.2 on Windows. The runner fixes the kernel working directory to that root and saves the executed notebook over its input on success. The venv commands use its interpreter explicitly.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name python3 --display-name "Python (NIDS research)"
```

The original runtime records Windows 11 and the core library versions. All 19 pinned distributions matched the installed inspection environment. These pins are the supplied package list. A clean-environment installation, Kaggle parity, peak RAM and wall-clock time remain unverified. The complete local package occupies 666 MB before new outputs or the Python environment. Allow disk space for another output tree and memory for in-memory DataFrames and fitted CPU estimators. The runner allows up to two hours **per cell**. [Runtime](../run_results/runtime.json) · [Package versions](../requirements-verified.txt)

**Saved-result replay.** Restore the four replay inputs described in the [data guide](../data/README.md). With those inputs at their expected paths, run:

```powershell
$env:NIDS_MODE = "replay"
.\.venv\Scripts\python.exe source/run_notebook.py
```

The default destination is `replay_results/`. Inspect its `execution_status.json`, `metric_verification.csv` and `paired_statistical_comparisons.csv` for the 44 historical metric groups and 28 conditional comparisons. The v3 external test and the controlled benchmark require train mode or their saved score exports for verification.

**Full training.** Restore the three verified training sample Parquets and the original RT-IoT2022 ZIP in the disposable copy. Use the [recorded splits, seeds, features and thresholds](METHODOLOGY.md). Set the mode explicitly:

```powershell
$env:NIDS_MODE = "train"
.\.venv\Scripts\python.exe source/run_notebook.py
```

The default destination is `run_results/`. Use a clean output directory for each run. `NIDS_OUTPUT` can override it, but the original verification and reporting scripts expect the result tree under `run_results/`. Set `NIDS_CLEAN_ROOT` for an alternative input root and `NIDS_SAVED` for alternative historical replay inputs. Use absolute paths for overrides. The source-calibrated threshold and RT scoring are embedded in this workflow. The external stage reuses fitted source models.

For interactive execution, start the notebook in the project root, select the same kernel, and use Restart Kernel and Run All. On Kaggle, attach an authorized full package privately, copy or extract it under `/kaggle/working`, and set the kernel working directory to the extracted folder containing `source/`, the notebook and data directories before running its cells. The original [archive README](../README.txt) gives the ZIP-specific setup. The sample-export environment differs from the local experiment environment.

## Output and provenance checks

From a Git checkout alone:

```sh
python -B -m unittest discover -s tests -v
```

For the pristine full delivery, first run `python source/check_package.py`. It expects the private archive's exact bytes, including its original notebook. All 685 listed files matched. The publication manifest covers the 134 distributed research files. Two local path prefixes in notebook stdout were removed for publication. The original cell sources, corrected source digest, execution counts and numerical results are preserved. The private session note, original UCI ZIP, fitted models and all Parquets remain outside Git.

With complete data and outputs, the following commands run in a **separate copy**. The first three update verification reports and the last rebuilds the paper tables, figures, HTML and editable report.

```sh
python source/verify_new_results.py
python source/verify_final.py
python source/check_splits.py
python source/report.py
```

During packaging the first validator recalculated 720 controlled and 120 native ablation rows from stored predictions. The second checked 180 external and 36 calibration rows, the identity of the external release, equality of source and external thresholds, and the protocol and notebook source hashes. The split audit checked 36 task/seed memberships. The original project separately records a same-machine fresh-kernel comparison of eight tables plus stored outputs, 526 files in all. Its `source/repeat_run.py` starts **another full training run**. [Verification record](../validation.json) · [Repeat-run evidence](../supplement/repeat_run.json)

An export of the Git index passed all six read-only tests without external files. After restoring the private Parquets and UCI archive into that disposable export, the original report builder reproduced all ten tables, five PNG figures, the HTML report and its editable text byte for byte. The original PDF figures were preserved and hash-checked.

The complete package matches all 590 hashes in the corrected output manifest, all 11 entries in the sample manifest and all 10 recorded training input hashes. Only 20 of 44 historical saved-results manifest entries are present. The missing entries are 22 old model checkpoints and two old figures. The four replay prerequisites are present in the full package. The upstream cleaned shards are identified by hash but unavailable for verification. The full historical cleaning notebook and a separate submission manuscript were not supplied.

## Computational amendment

After the first v3 external scores were known, the authors fixed the order in which Random Forest tree probabilities were summed during prediction by setting `n_jobs=1` after fitting. The fits, splits, endpoints and threshold-selection policy remained unchanged. Applying that policy to the corrected scores changed the numerical threshold in eight rows relative to the first v3 run (three controlled benchmark, four native ablation and one external evaluation). Confusion counts changed in two rows, both in the controlled benchmark. The prespecified LightGBM contrast was unaffected. [Row-by-row RF comparison](../supplement/rf_implementation_comparison.csv) · [Original amendment and timing](../protocol/computational_amendment.json) · [Before-training record](../protocol/before_training.json)

Recalculation validates the exported arithmetic. The external result covers one additional laboratory collection. Independent reproduction by another team and validation on production traffic remain open.
