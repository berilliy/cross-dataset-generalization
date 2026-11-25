"""Replay the complete training notebook in a fresh kernel and compare numerical outputs exactly."""
from pathlib import Path
from tempfile import mkdtemp, gettempdir
from hashlib import file_digest
import json, os, shutil, sys
import nbformat, pandas as pd
from nbclient import NotebookClient

root=Path(__file__).resolve().parents[1]
reference=root/"run_results"


def compare(other):
    rows=[]
    for name in ["controlled_benchmark","native_ablation","strict_temporal_metrics","unseen_domain_metrics",
                 "external_metrics","calibration_sensitivity","paired_statistical_comparisons",
                 "conditional_cluster_confidence_intervals"]:
        path=name+".csv"
        a=pd.read_csv(reference/path,float_precision="round_trip")
        b=pd.read_csv(other/path,float_precision="round_trip")
        pd.testing.assert_frame_equal(a,b,check_exact=True)
        rows.append(dict(path=path,rows=len(a),check="exact numeric and text table",status="PASS"))
    prediction_paths=[]
    for name in ["benchmark_predictions","native_predictions","external_scores"]:
        filelist=sorted(f.relative_to(reference) for f in (reference/name).glob("*.parquet"))
        assert filelist==sorted(f.relative_to(other) for f in (other/name).glob("*.parquet"))
        prediction_paths.extend(filelist)
    assert len(prediction_paths)==510
    prediction_paths+=list(map(Path,["strict_temporal_predictions.parquet","unseen_predictions.parquet","external_target.parquet","benchmark_membership.parquet"]))
    prediction_paths+=sorted(f.relative_to(reference) for f in reference.glob("calibration_scores*.parquet"))
    for rel in prediction_paths:
        base=reference/rel
        with base.open("rb") as fa,(other/rel).open("rb") as fb:
            identical=file_digest(fa,"sha256").digest()==file_digest(fb,"sha256").digest()
        if not identical:
            a=pd.read_parquet(base)
            b=pd.read_parquet(other/rel)
            pd.testing.assert_frame_equal(a,b,check_exact=True)
        rows.append(dict(path=rel.as_posix(),rows=None,check="identical bytes" if identical else "identical stored values",status="PASS"))
    pd.DataFrame(rows).to_csv(root/"supplement/repeat_run_comparison.csv",index=False)
    summary=dict(status="PASS",separate_fresh_kernel=True,independent_team=False,
                 exact_tables=8,prediction_files_by_fit=510,all_files_compared=len(rows),
                 same_machine_and_dependencies=True,repeat_output=str(other.name))
    (root/"supplement/repeat_run.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    if len(sys.argv)==3 and sys.argv[1]=="--compare":
        compare(Path(sys.argv[2]).resolve())
    else:
        other=Path(mkdtemp(prefix="nids_research_repeat_"))
        print("Second output directory:",other,flush=True)
        os.environ["NIDS_MODE"]="train"
        os.environ["NIDS_OUTPUT"]=str(other)
        n=nbformat.read(root/"notebook3ec3218ec0_corrected.ipynb",4)
        def progress(cell,cell_index,execute_reply):
            print("Repeat cell",cell_index,execute_reply["content"]["status"],flush=True)
        NotebookClient(n,timeout=7200,kernel_name="python3",resources={"metadata":{"path":str(root)}},
                       on_cell_executed=progress).execute()
        compare(other)
        # Keep failed runs for diagnosis; a successful repeat is fully represented by the reference.
        if other.resolve().parent==Path(gettempdir()).resolve() and other.name.startswith("nids_research_repeat_"):
            shutil.rmtree(other)

