"""Separate arithmetic path for the external test and calibration outcomes."""
from pathlib import Path
from hashlib import sha256
import json, math
from zipfile import ZipFile
import numpy as np
import pandas as pd
import nbformat
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
R = ROOT/"run_results"


def verify_metrics(row, y, scores):
    pred = scores >= row.threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    assert (tn,fp,fn,tp)==(row.tn,row.fp,row.fn,row.tp) and len(y)==row.n
    rec=tp/(tp+fn)
    fpr=fp/(fp+tn)
    values={"fpr":fpr,"recall":rec,"precision":tp/(tp+fp) if tp+fp else 0.,
            "accuracy":(tn+tp)/len(y),"f1":f1_score(y,pred,zero_division=0),
            "macro_f1":f1_score(y,pred,average="macro",zero_division=0),
            "roc_auc":roc_auc_score(y,scores),"average_precision":average_precision_score(y,scores),
            "f1_at_20pct_attack":2*.2*rec/(.2*(1+rec)+.8*fpr)}
    for name,number in values.items():
        assert np.isclose(number,getattr(row,name),rtol=0,atol=1e-12),name


def verify():
    get=lambda name: pd.read_csv(R/(name+".csv"),float_precision="round_trip")
    raw=ROOT/"external_data/rt-iot2022.zip"
    with ZipFile(raw) as z, z.open("RT_IOT2022") as f:
        original=pd.read_csv(f,usecols=["Attack_type"])
    truth=pd.read_parquet(R/"external_target.parquet")
    assert np.array_equal(truth.row_number,np.arange(len(original))) and len(truth)==123117
    assert original.Attack_type.equals(truth.attack_label)
    y=(~original.Attack_type.isin(["MQTT_Publish","Thing_Speak","Wipro_bulb","Amazon-Alexa"])).astype("int8")
    assert y.equals(truth.target)
    checks=[]
    ext=get("external_metrics"); b=get("controlled_benchmark")
    anchor=b[b.target.eq("CICIOT2023") & b.scenario.ne("in_domain")]
    keys=["sources","model","seed","variant","policy"]
    joined=ext.merge(anchor[keys+["threshold"]],on=keys,validate="one_to_one",suffixes=("","_source"))
    assert len(joined)==len(ext)==len(anchor)==180
    assert np.array_equal(joined.threshold,joined.threshold_source)
    for key,g in ext.groupby("key"):
        s=pd.read_parquet(R/"external_scores"/(key+".parquet"))
        assert s.row_number.equals(truth.row_number) and np.isfinite(s.score).all() and len(g)==2
        for row in g.itertuples(index=False):
            verify_metrics(row,y,s.score)
            checks.append(dict(family="external",condition=key,policy=row.policy,rows=len(y)))
    c=get("calibration_sensitivity")
    results=get("strict_temporal_metrics")
    preds=pd.read_parquet(R/"strict_temporal_predictions.parquet")
    for (ds,name),g in c.groupby(["dataset","model"]):
        cal=pd.read_parquet(R/f"calibration_scores_{ds}_{name}.parquet")
        tests=preds[(preds.dataset==ds)&(preds.model==name)&(preds.policy=="default_0.5")]
        assert cal._row_key.is_unique and tests.row_key.is_unique and not set(cal._row_key)&set(tests.row_key)
        midday=cal._event_time.min()+(cal._event_time.max()-cal._event_time.min())/2
        full=results[(results.dataset==ds)&(results.model==name)&(results.policy=="source_cal_fpr_1pct")].iloc[0]
        for row in g.itertuples(index=False):
            w={"full":cal,"early":cal[cal._event_time<midday],"late":cal[cal._event_time>=midday]}[row.window]
            neg=w[w.target.eq(0)].score.to_numpy()
            assert row.status=="evaluated" and row.calibration_n==len(w) and row.benign_n==len(neg)
            assert row.threshold==np.nextafter(np.sort(neg)[math.ceil((1-row.alpha)*len(neg))-1],np.inf)
            assert abs(row.calibration_fpr-(neg>=row.threshold).mean())<1e-12
            if row.window=="full" and row.alpha==.01:
                assert row.threshold==full.threshold and row.fp==full.fp and row.tp==full.tp
            verify_metrics(row,tests.target,tests.score)
            checks.append(dict(family="calibration",condition=f"{ds}/{name}/{row.window}",policy=row.alpha,rows=len(tests)))
    assert len(c)==36 and len(checks)==216
    old=get("independent_export_verification")
    paired=get("paired_statistical_comparisons")
    assert len(old)==840 and old.status.eq("PASS").all() and len(paired)==28
    assert paired.valid_null_resamples.min()==1000 and paired.valid_macro_bootstrap.min()==1000
    assert len(get("metric_verification"))==44
    n=nbformat.read(ROOT/"notebook3ec3218ec0_corrected.ipynb",4)
    stamp=json.loads((ROOT/"protocol/before_training.json").read_text())
    recorded=json.loads((ROOT/"protocol/freeze_record.json").read_text())
    assert sha256((ROOT/"protocol/v3_protocol.json").read_bytes()).hexdigest()==stamp["protocol"]==recorded["sha256"]
    assert sha256((ROOT/"source/external_eval.py").read_bytes()).hexdigest()==stamp["external_module"]
    revised=json.loads((ROOT/"protocol/computational_amendment.json").read_text())
    current='\n'.join(cell.source for cell in n.cells)
    assert sha256(current.encode()).hexdigest()==revised["notebook_source"]
    import re
    original, amendments=re.subn(r"(?m)^([ ]*)# Accumulate tree probabilities in tree order; ties near the quantile are decision-relevant\.\n\1if name == 'RandomForest':\n\1    model\.named_steps\['model'\]\.set_params\(n_jobs=1\)\n","",current)
    assert amendments==2 and sha256(original.encode()).hexdigest()==stamp["notebook_source"]
    assert sha256(raw.read_bytes()).hexdigest()==stamp["input_release"]
    code=[cell for cell in n.cells if cell.cell_type=="code"]
    assert [cell.execution_count for cell in code]==list(range(1,len(code)+1))
    assert not any(o.output_type=="error" for cell in code for o in cell.outputs)
    assert n.cells[0].source.splitlines()[0]=="# "+json.loads((ROOT/"protocol/v3_protocol.json").read_text(encoding="utf-8-sig"))["title"]
    pd.DataFrame(checks).assign(status="PASS").to_csv(ROOT/"supplement/verification_new.csv",index=False)
    out=dict(real_train_run_all="PASS",completed_code_cells=len(code),error_outputs=0,
             benchmark_rows=720,native_ablation_rows=120,external_rows_checked=180,
             calibration_rows_checked=36,existing_benchmark_native_checked=len(old),
             additional_strict_and_exploratory_metric_groups=44,
             frozen_source_threshold_equality_for_external=True,local_freeze_verified=True,
             paired_conditional_tests=28,external_records=len(truth),
             external_testbeds=1,independent_team_replication=False)
    (ROOT/"validation.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))


if __name__=="__main__":
    verify()

