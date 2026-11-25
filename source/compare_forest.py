"""Record the impact of deterministic RF probability accumulation on known results."""
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[1]
out=[]
for table,keys in [
    ("controlled_benchmark",["target","sources","seed","variant","model","policy"]),
    ("external_metrics",["sources","seed","variant","model","policy"]),
    ("native_ablation",["dataset","seed","variant","model","policy"])]:
    before=pd.read_csv(root/"supplement"/("initial_parallel_rf_"+table+".csv"),float_precision="round_trip")
    after=pd.read_csv(root/"run_results"/(table+".csv"),float_precision="round_trip").query("model=='RandomForest'")
    cols=["threshold","macro_f1","fpr","recall","tn","fp","fn","tp"]
    pair=before[keys+cols].merge(after[keys+cols],on=keys,suffixes=("_initial","_final"),validate="one_to_one")
    assert len(pair)==len(before)==len(after)
    pair["table"]=table
    pair["decisions_changed"]=(pair.tp_initial!=pair.tp_final)|(pair.fp_initial!=pair.fp_final)
    out.append(pair)
summary=pd.concat(out,ignore_index=True)
summary.to_csv(root/"supplement/rf_implementation_comparison.csv",index=False)
print(summary.groupby(["table","policy"]).decisions_changed.agg(["count","sum"]).to_string())

