# Initial release checklist

## Prepared locally

- The corrected v3 package is the scientific reference. The original delivery stays outside the Git repository. Its 685 manifest entries match.
- The publication copy carries the original scripts, protocols, amendment, report, figures, CSV tables and validation evidence at their original relative paths. A session-specific provenance note remains private. Its acquisition details are documented in the data guide.
- The publication manifest records research-file bytes, including the output-only notebook path redactions. Git attributes preserve historical line endings. CSVs are excluded by default and individual validated tables and manifests are allowed explicitly.
- Before tagging any revision, run `python -B -m unittest discover -s tests -v` and inspect the staged inventory with `git diff --cached --stat`. The [reproducibility guide](REPRODUCIBILITY.md) records the numerical checks and limits of the data available in Git.

## Owner decisions before public release

- Confirm author names and order in [CITATION.cff](../CITATION.cff). Its empty author list is a draft and does not satisfy the CFF 1.2.0 release schema. Add publication metadata only when assigned.
- Review the MIT proposal below and confirm rights to distribute the report, figures and supplementary material.
- Approve the data availability statement. If exact samples or score exports may be shared separately, supply the authorized access route. Otherwise retain the documented access limitations.
- Approve the public GitHub destination and release under `nids-cross-dataset-generalization`. Record the resulting tag or commit in the paper citation.

## Proposed MIT license for project software

**Pending owner approval.** The proposed scope is the project-owned Python code in `source/`, the project-owned code cells in the research notebook and the project-owned tests in `tests/`. The proposal uses the standard MIT text, which permits reuse and redistribution subject to retaining the copyright and permission notices and includes a warranty disclaimer. [Standard MIT license text](https://opensource.org/license/mit)

UNSW-NB15, Network ToN-IoT, CICIoT2023 and RT-IoT2022 retain their respective dataset terms. Third-party software and material retain their own licenses and attribution requirements. The report, figures, tables, notebook outputs and supplementary research materials are outside the proposed software grant. [Dataset provenance](DATASETS.md) · [UCI attribution and release terms](../external_data/release.json)

To approve, confirm the software scope, copyright holder and year for the notice. After explicit approval, the standard text can be placed in a root `LICENSE` file and its scope recorded in the README. The current repository carries no MIT grant.
