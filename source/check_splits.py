"""Cross-check identities, partitions and common feature values independently of notebook hashes."""
from pathlib import Path
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]
r=root/"run_results"
lookup=[]
summaries=[]
for ds, dur in [("UNSW_NB15","dur"),("TON_IOT","duration"),("CICIOT2023","flow_duration")]:
    cic=ds=="CICIOT2023"
    cols=["TCP","UDP","ICMP"] if cic else ["proto"]
    raw=pd.read_parquet(root/f"training_samples/{ds}/data/sample.parquet",columns=[dur,"target","source_file","source_row"]+cols)
    val=pd.to_numeric(raw[dur],errors="raise")
    if not cic: val=val.astype("float32")
    x=pd.DataFrame({"log_duration":np.log1p(val.where(val>=0)).astype("float64")})
    for k in ("tcp","udp","icmp"):
        v=raw[k.upper()].gt(.5) if cic else (raw.proto.str.lower().str.strip().str.contains("icmp",na=False) if k=="icmp" else raw.proto.str.lower().str.strip().eq(k))
        x[k]=v.astype("int8")
    x["other"]=x[["tcp","udp","icmp"]].sum(axis=1).eq(0).astype("int8")
    sep=":" if cic else "|"
    key=ds+sep+raw.source_file.astype(str)+sep+raw.source_row.astype(str)
    h=pd.util.hash_pandas_object(x,index=False).to_numpy(dtype="uint64")
    ref=pd.DataFrame({"_row_key":key,"canon":h,"target_label":raw.target})
    actual=pd.read_parquet(r/f"{ds}_benchmark_splits.parquet")
    partitioned=actual.merge(ref,on="_row_key",validate="one_to_one")
    assert len(actual)==len(partitioned)==len(raw)
    for name,part in partitioned.groupby("partition"):
        summaries.append(dict(domain=ds,split=name,n=len(part),attacks=int(part.target_label.sum()),
                              attack_fraction=part.target_label.mean(),unique_keys=part.canon.nunique()))
    lookup.append(ref[["_row_key","canon"]])
ids=pd.concat(lookup,ignore_index=True)
members=pd.read_parquet(r/"benchmark_membership.parquet")
members=members.merge(ids,on="_row_key",validate="many_to_one",how="left")
assert members.canon.notna().all()
checks=[]
for (target,sources,seed),g in members.groupby(["target","sources","seed"]):
    sets={p:set(z.canon) for p,z in g.groupby("partition")}
    duplicate_rows=g.duplicated("_row_key").sum()
    collisions=[len(sets[a]&sets[b]) for a,b in [("train","calibration"),("train","test"),("calibration","test")]]
    assert not duplicate_rows and not any(collisions), (target,sources,seed,collisions)
    checks.append(dict(target=target,sources=sources,seed=seed,duplicate_row_ids=duplicate_rows,
                       train_cal_groups=collisions[0],train_test_groups=collisions[1],cal_test_groups=collisions[2]))
assert len(checks)==36
pd.DataFrame(checks).to_csv(root/"supplement/canonical_split_checks.csv",index=False)
pd.DataFrame(summaries).to_csv(root/"supplement/partition_composition.csv",index=False)
print("Verified all 36 task/seed memberships at common numeric precision.")

