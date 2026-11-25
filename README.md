# Robust Cross-Dataset Generalization of Machine Learning-Based Network Intrusion Detection Under Temporal and Distributional Shift

A network intrusion detector can separate attacks from benign traffic on one testbed and still raise frequent false alarms on another. This study follows the same classifiers across networks and calendar periods. It asks how much performance survives a change of source, what a constant classifier would achieve on the same traffic, and whether source-calibrated thresholds retain their meaning at the destination.

The repository accompanies the corrected v3 analysis. Start with the [illustrated research report](research.html) (download and open locally), the [figure and table index](docs/PROJECT_STRUCTURE.md), or the [reproducibility guide](docs/REPRODUCIBILITY.md).

## Research questions

| | Question |
|---|---|
| RQ1 | How much does classification performance change when moving between datasets? |
| RQ2 | Which models retain an advantage over a simple decision rule? |
| RQ3 | Are measured feature shifts associated with the loss of performance? |
| RQ4 | Does a threshold calibrated on earlier traffic transfer to later days? |
| RQ5 | Does combining sources improve performance at a fixed training budget? |

## Datasets and design

| Dataset | Controlled benchmark | UTC calendar evaluation | External evaluation |
|---|---|---|---|
| UNSW-NB15 | Training source and held-out target | Jan 22 / Jan 23 / Feb 18, 2015 | Training source |
| Network ToN-IoT | Training source and held-out target | Through Apr 24 / Apr 25 / Apr 26–27, 2019 | Training source |
| CICIoT2023 | Training source and held-out target | — | — |
| RT-IoT2022 | — | — | All 123,117 rows, evaluation only |

For each of the first three targets, the controlled benchmark compares within-domain training, each alternative source alone, and the two alternatives pooled. Each fit uses 60,000 training rows. Full common-feature vectors determine fixed train, calibration and test groups. Seeds 42, 7 and 2026 vary training within that design. Majority predicts from the unweighted training class frequencies. The learned models are logistic regression, Random Forest, histogram gradient boosting and LightGBM.

Two representations use the same splits. Both encode TCP, UDP, ICMP and other traffic. The larger representation adds log-transformed duration. The study evaluates a fixed 0.5 threshold and a threshold fitted to the benign tail on source calibration data. The external test reuses the UNSW, ToN and pooled fits from the CIC target loop. It uses neither RT labels for calibration nor RT rows for training. [Methodology](docs/METHODOLOGY.md) · [Frozen external protocol](protocol/v3_protocol.json)

## Findings

On RT-IoT2022, pooling substantially improved on UNSW alone but fell below ToN alone. The prespecified comparison uses LightGBM, protocol plus duration, threshold 0.5 and means across the three seeds. [External summary](tables/external_summary.csv) · [Paired pooling contrasts](tables/external_pooling.csv)

| Training source | Macro-F1 | False positive rate | Attack recall |
|---|---:|---:|---:|
| UNSW-NB15 | 0.0949 | 0.5399 | 0.0536 |
| Network ToN-IoT | 0.6158 | 0.4518 | 0.8238 |
| Both sources | 0.5580 | 0.6341 | 0.8212 |

- In four of nine cross-domain conditions, all four learned models scored below Majority on mean Macro-F1 with duration at threshold 0.5. The direction of the duration effect also depended on the source and target. [Majority comparisons](tables/majority_summary.csv) · [Duration ablation](tables/duration_summary.csv)
- Across nine transfer conditions, mean feature shift did not consistently rank performance loss. The Spearman associations differed across models and threshold policies. [Shift correlations](tables/shift_correlations.csv)
- In the UNSW calendar test, full-window calibration at a 1% benign-tail budget gave logistic regression a calibration FPR of 0.99% and a later test FPR of 15.71%, with 99.99% attack recall. On ToN, LightGBM under the same policy detected just 3 of 60,284 later attacks. [Calibration sensitivity](run_results/calibration_sensitivity.csv)

## Reproducing the analysis

From the checkout root, run `python -B -m unittest discover -s tests -v` for read-only checks using Python 3.11 or later. These checks use the published tables and manifests. For notebook execution, use Python 3.14.2, install `requirements.txt`, and work in a disposable copy. The notebook runner writes its executed notebook back to disk and defaults to training. Exact sample files, replay predictions and the UCI ZIP are separate inputs. [Installation, modes and verification](docs/REPRODUCIBILITY.md) · [Data acquisition and layout](data/README.md)

The original authors recorded two corrected runs on one machine. A prediction-order amendment for Random Forest was made **after the first v3 external scores were known**. Both the timing and the superseded results are retained. [Amendment](protocol/computational_amendment.json) · [Measured differences](supplement/rf_implementation_comparison.csv)

## Repository guide

The notebook and report live at the root to preserve their working paths. `source/` holds execution and verification scripts, `protocol/` the frozen design, `run_results/` the corrected run summaries, and `tables/` and `figures/` the paper exhibits. `supplement/` holds the methods, audits and repeat-run evidence. `saved_results/` describes historical comparisons. [Full directory guide](docs/PROJECT_STRUCTURE.md)

## Limits of inference

Flow exporters can give duration different meanings. Equal protocol or duration values need not identify the same kind of event. Large groups of repeated vectors and unequal attack prevalence affect the splits and the usefulness of Majority. Only two datasets support the reported day-level comparison. Flows can share sessions, devices and capture conditions, and three seeds on a single target do not provide three independent networks. The original three targets were previously inspected. RT adds one external laboratory testbed. Independent production-network validation remains open.

## Citation, licensing and credits

Cite the exact title above and identify the repository revision used. Author attribution awaits owner confirmation in the [citation file](CITATION.cff). The software license also awaits owner approval. Dataset licenses apply separately. [Release checklist](docs/RELEASE.md)

The work uses datasets by Nour Moustafa and Jill Slay (UNSW-NB15), Nour Moustafa and collaborators (Network ToN-IoT), Euclides Carlos Pinto Neto and colleagues (CICIoT2023), and B. S. Sharmila and Rohini Nagapadma (RT-IoT2022). The report situates its transfer experiments alongside the cross-dataset studies by Cantone et al. and Iwanowski et al. [Dataset sources](docs/DATASETS.md) · [Research bibliography](supplement/references.bib) · [Literature comparison](literature_comparison.csv)
