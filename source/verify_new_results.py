from pathlib import Path
import pandas as pd, numpy as np, json, nbformat
from sklearn.metrics import roc_auc_score, average_precision_score
D=Path(__file__).resolve().parents[1];R=D/'run_results'
checks=[]
for family,filename,folder in [('core','controlled_benchmark.csv','benchmark_predictions'),('native','native_ablation.csv','native_predictions')]:
    metrics=pd.read_csv(R/filename,float_precision='round_trip')
    for f in sorted((R/folder).glob('*.parquet')):
        parts=f.stem.split('__')
        if family=='core':
            target,sources,seed,variant,model=parts
            mask=metrics.target.eq(target)&metrics.sources.eq(sources)&metrics.seed.eq(int(seed))&metrics.variant.eq(variant)&metrics.model.eq(model)
        else:
            target,variant,seed,model=parts
            mask=metrics.dataset.eq(target)&metrics.seed.eq(int(seed))&metrics.variant.eq(variant)&metrics.model.eq(model)
        z=pd.read_parquet(f);assert z._row_key.is_unique
        selected=metrics[mask];assert len(selected)==2
        assert np.isfinite(z.score).all() and z.target.isin([0,1]).all()
        for row in selected.itertuples(index=False):
            assert len(z)==row.n
            if row.policy=='source_cal_fpr_1pct':assert np.isclose(row.threshold,z.threshold.iloc[0],atol=0,rtol=0)
            y=z.target.to_numpy();pred=z.score.to_numpy()>=row.threshold
            tn=int(((y==0)&~pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum());tp=int(((y==1)&pred).sum())
            assert (tn,fp,fn,tp)==(row.tn,row.fp,row.fn,row.tp)
            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
            fb=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0
            rec=tp/(tp+fn);fpr=fp/(fp+tn)
            standardized=2*.2*rec/(.2*(1+rec)+.8*fpr)
            for name,value in [('macro_f1',(f1+fb)/2),('f1',f1),('fpr',fpr),('recall',rec),('f1_at_20pct_attack',standardized),('roc_auc',roc_auc_score(y,z.score)),('average_precision',average_precision_score(y,z.score))]:
                assert np.isclose(value,getattr(row,name),atol=1e-12), (f.name,row.policy,name)
            checks.append(dict(family=family,file=f.name,policy=row.policy,n=len(z),status='PASS'))
    assert len(metrics)==sum(z['family']==family for z in checks)
pd.DataFrame(checks).to_csv(R/'independent_export_verification.csv',index=False)
print('NEW_EXPORTS_VERIFIED',len(checks),flush=True)
assert len(checks)==840
# Exact split identifiers and non-overlap for the new cross-domain benchmark.
members=pd.read_parquet(R/'benchmark_membership.parquet')
for keys,z in members.groupby(['target','sources','seed']):
    groups={k:set(g._group) for k,g in z.groupby('partition')}
    assert not groups['train']&groups['calibration'] and not groups['train']&groups['test'] and not groups['calibration']&groups['test']
    assert len(z[z.partition.eq('train')])==60000
for target,z in members[members.partition.eq('test')].groupby('target'):
    sets=[set(g._row_key) for _,g in z.groupby(['sources','seed'])]
    assert all(s==sets[0] for s in sets)
n=nbformat.read(D/'notebook3ec3218ec0_corrected.ipynb',as_version=4);nbformat.validate(n)
assert n.cells[0].source.splitlines()[0]=='# '+json.loads((D/'protocol/v2_protocol.json').read_text())['title']
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in n.cells if c.cell_type=='code')
print('SPLITS_AND_TRAIN_RUN_ALL_VERIFIED',flush=True)
