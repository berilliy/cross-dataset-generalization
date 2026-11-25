"""Descriptive measurement QA; no outcome labels are read."""
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd


def audit(root):
    out=root/"supplement"
    summaries=[]
    for domain,col in [("UNSW_NB15","dur"),("TON_IOT","duration"),("CICIOT2023","flow_duration")]:
        columns=[col]
        if domain=="CICIOT2023":columns.extend(["TCP","UDP","ICMP"])
        d=pd.read_parquet(root/f"training_samples/{domain}/data/sample.parquet",columns=columns)
        vals=pd.to_numeric(d[col],errors="coerce")
        record={"dataset":domain,"column":col,"n":len(d),"missing":int(vals.isna().sum()),"negative":int(vals.lt(0).sum()),
                "zero":int(vals.eq(0).sum()),"median_native_units":vals.median(),"p95_native_units":vals.quantile(.95)}
        if domain=="CICIOT2023":
            flags=d[["TCP","UDP","ICMP"]]
            record.update(protocol_flag_missing_rows=int(flags.isna().any(axis=1).sum()),
                          protocol_flag_fractional_rows=int(((flags>0)&(flags<1)).any(axis=1).sum()),
                          no_majority_protocol=int(flags.le(.5).all(axis=1).sum()))
        summaries.append(record)
    with ZipFile(root/"external_data/rt-iot2022.zip") as z,z.open("RT_IOT2022") as f:
        d=pd.read_csv(f,usecols=["flow_duration","flow_pkts_per_sec","fwd_pkts_tot","bwd_pkts_tot"])
    v=d.flow_duration
    summaries.append(dict(dataset="RT-IoT2022",column="flow_duration",n=len(d),missing=int(v.isna().sum()),
                          negative=int(v.lt(0).sum()),zero=int(v.eq(0).sum()),median_native_units=v.median(),
                          p95_native_units=v.quantile(.95)))
    pd.DataFrame(summaries).to_csv(out/"duration_profiles.csv",index=False)
    # N / (packets per second) is in seconds. Absolute tolerance reflects CSV rounding of short flows.
    use=v.gt(0)&d.flow_pkts_per_sec.gt(0)
    implied=(d.fwd_pkts_tot[use]+d.bwd_pkts_tot[use])/d.flow_pkts_per_sec[use]
    observed=v[use]
    diff=(implied-observed).abs()
    tol=np.maximum(1e-6,.01*observed)
    row=dict(total=len(d),eligible=int(use.sum()),excluded=int((~use).sum()),agree=int((diff<=tol).sum()),
             median_ratio_seconds_over_reported=float((implied/observed).median()),
             median_abs_error_seconds=float(diff.median()),tolerance="max(1e-6 seconds, 1% of reported duration)",
             status="retrospective unit-consistency diagnostic, no calibration or tuning")
    pd.DataFrame([row]).to_csv(out/"rt_duration_identity.csv",index=False)
    return row


if __name__=="__main__":
    print(audit(Path(__file__).resolve().parents[1]))

