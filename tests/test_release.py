"""Read-only checks for the publication snapshot, using only the standard library."""
import csv
import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = "notebook3ec3218ec0_corrected.ipynb"


def rows(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, strict=True))


def load_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def digest(path):
    with (ROOT / path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class PublicationChecks(unittest.TestCase):
    def test_publication_hashes_and_provenance(self):
        package = {r["file"]: r["sha256"] for r in rows("package_sha256.csv")}
        release = rows("docs/release_sha256.csv")
        self.assertEqual(len({r["file"] for r in release}), 134)
        for record in release:
            path = record["file"]
            with self.subTest(path=path):
                self.assertEqual((ROOT / path).stat().st_size, int(record["bytes"]))
                self.assertEqual(digest(path), record["sha256"])
                if path not in (NOTEBOOK, "package_sha256.csv"):
                    self.assertEqual(record["sha256"], package[path])

    def test_protocol_and_executed_notebook(self):
        notebook = load_json(NOTEBOOK)
        protocol = load_json("protocol/v3_protocol.json")
        amendment = load_json("protocol/computational_amendment.json")
        before = load_json("protocol/before_training.json")
        title = protocol["title"]
        self.assertEqual(title, load_json("protocol/v2_protocol.json")["title"])
        self.assertEqual((ROOT / "README.md").read_text("utf-8").splitlines()[0], "# " + title)
        self.assertEqual(digest("protocol/v3_protocol.json"), before["protocol"])
        self.assertEqual(digest("source/external_eval.py"), before["external_module"])
        self.assertEqual(protocol["seeds"], [42, 7, 2026])
        cells = notebook["cells"]
        sources = ["".join(cell["source"]) for cell in cells]
        self.assertEqual(sources[0].splitlines()[0], "# " + title)
        self.assertEqual(hashlib.sha256("\n".join(sources).encode()).hexdigest(),
                         amendment["notebook_source"])
        code = [c for c in cells if c["cell_type"] == "code"]
        self.assertEqual([c["execution_count"] for c in code], list(range(1, 15)))
        for cell in code:
            self.assertFalse(any(o["output_type"] == "error" for o in cell["outputs"]))

    def test_research_file_formats(self):
        for record in rows("docs/release_sha256.csv"):
            path = Path(record["file"])
            with self.subTest(path=path):
                if path.suffix == ".json":
                    load_json(path)
                if path.suffix == ".csv":
                    with (ROOT / path).open(encoding="utf-8-sig", newline="") as stream:
                        reader = csv.reader(stream, strict=True)
                        header = next(reader)
                        if path.name in {"benchmark_seed_summary.csv", "native_seed_summary.csv"}:
                            # pandas exports metric/statistic columns and named index levels.
                            statistic, index_names = next(reader), next(reader)
                            header = list(zip(header, statistic, index_names, strict=True))
                        self.assertEqual(len(header), len(set(header)))
                        self.assertTrue(all(len(row) == len(header) for row in reader))

    def test_confusion_counts_and_metrics(self):
        sizes = {"controlled_benchmark": 720, "native_ablation": 120,
                 "external_metrics": 180, "calibration_sensitivity": 36}
        for table, size in sizes.items():
            records = rows(f"run_results/{table}.csv")
            self.assertEqual(len(records), size)
            for i, row in enumerate(records):
                with self.subTest(table=table, record=i):
                    tp, tn, fp, fn = (int(row[k]) for k in ("tp", "tn", "fp", "fn"))
                    self.assertEqual(tp + tn + fp + fn, int(row["n"]))
                    self.assertAlmostEqual(tp / (tp + fn), float(row["recall"]), places=12)
                    self.assertAlmostEqual(fp / (tn + fp), float(row["fpr"]), places=12)
                    macro = tp / (2 * tp + fp + fn) + tn / (2 * tn + fp + fn)
                    self.assertAlmostEqual(macro, float(row["macro_f1"]), places=12)

    def test_external_threshold_transfer(self):
        keys = ["sources", "seed", "variant", "model", "policy"]
        def key(row):
            return tuple(row[k] for k in keys)
        anchors = {key(r): r for r in rows("run_results/controlled_benchmark.csv")
                   if r["target"] == "CICIOT2023" and r["scenario"] != "in_domain"}
        external = rows("run_results/external_metrics.csv")
        self.assertEqual(len(anchors), 180)
        self.assertEqual(set(anchors), {key(r) for r in external})
        for row in external:
            self.assertEqual(float(row["threshold"]), float(anchors[key(row)]["threshold"]))
            self.assertEqual(int(row["n"]), 123117)

    def test_documentation_links(self):
        paths = [ROOT / "README.md", ROOT / "data/README.md", *ROOT.glob("docs/*.md")]
        for path in paths:
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text("utf-8")):
                if not target.startswith(("http://", "https://", "#")):
                    self.assertTrue((path.parent / target).exists(), f"{path.name} -> {target}")
