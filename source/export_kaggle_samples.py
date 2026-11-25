# Reproducible, label-independent research sample export. Run as one Kaggle cell.
from pathlib import Path
import json, hashlib, shutil, sys
import numpy as np
import pandas as pd
from IPython.display import display, FileLink
roots=sorted({f.parent.parent for f in Path('/kaggle/input').rglob('UNSW_NB15/manifest.json') if (f.parent.parent/'TON_IOT/data').is_dir() and (f.parent.parent/'CICIOT2023/data').is_dir()})
local=Path('/kaggle/working/nids_clean_v1')
if all((local/d/'data').is_dir() for d in ['UNSW_NB15','TON_IOT','CICIOT2023']):roots=[local]
assert len(roots)==1, f'Expected one complete clean root, found {roots}'
root=roots[0]; dest=Path('/kaggle/working/nids_research_samples_v2');dest.mkdir(exist_ok=True)
rates={'UNSW_NB15':.07,'TON_IOT':.015,'CICIOT2023':.025}
cats={'UNSW_NB15':['proto','state','service'],'TON_IOT':['proto','conn_state','service']}
times={'UNSW_NB15':'meta_Stime','TON_IOT':'meta_ts'}
deny={'target','label','attack_cat','attack_label','type','id','stime','ltime','ts','timestamp','srcip','dstip','src_ip','dst_ip'}
inputs=[]; summaries=[]
for ds,rate in rates.items():
    schema=json.loads((root/ds/'schema.json').read_text())
    if ds in cats:
        nums=[c for c in schema['numeric_columns'] if c.lower() not in deny and not c.lower().startswith('meta_')]
        cc=[c for c in cats[ds] if c in schema['categorical_columns']]
        cols=list(dict.fromkeys(nums+cc+[times[ds],'target','attack_label','source_file','source_row']))
        schema['numeric_columns']=nums;schema['categorical_columns']=cc
    else:
        cols=['flow_duration','TCP','UDP','ICMP','target','attack_label','source_file','source_row']
        schema['numeric_columns']=['flow_duration','TCP','UDP','ICMP'];schema['categorical_columns']=[]
    chunks=[]
    for f in sorted((root/ds/'data').glob('*.parquet')):
        hfile=hashlib.sha256()
        with f.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):hfile.update(block)
        inputs.append(dict(dataset=ds,file=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=hfile.hexdigest()))
        z=pd.read_parquet(f,columns=cols)
        h=pd.util.hash_pandas_object(z[['source_file','source_row']],index=False).to_numpy('uint64')
        z=z.loc[(h & np.uint64(0xffffffff))/float(2**32)<rate].copy()
        if len(z):chunks.append(z)
    z=pd.concat(chunks,ignore_index=True)
    assert not z[['source_file','source_row']].duplicated().any()
    target=dest/ds; (target/'data').mkdir(parents=True,exist_ok=True)
    z.to_parquet(target/'data'/'sample.parquet',index=False)
    (target/'schema.json').write_text(json.dumps(schema,indent=2))
    shutil.copy2(root/ds/'manifest.json',target/'manifest.json')
    summaries.append(dict(dataset=ds,rows=len(z),sample_rate=rate,columns=cols))
    print('EXPORTED',ds,len(z),flush=True)
pd.DataFrame(inputs).to_csv(dest/'upstream_parquet_sha256.csv',index=False)
(dest/'sample_export.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,root=str(root),sampling='pandas hash(source_file,source_row), low32 / 2**32 < fixed rate; no label selection',datasets=summaries),indent=2))
files=[]
for f in sorted(dest.rglob('*')):
    if f.is_file():files.append(dict(file=str(f.relative_to(dest)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
pd.DataFrame(files).to_csv(dest/'sample_sha256.csv',index=False)
archive=shutil.make_archive('/kaggle/working/NIDS_research_samples_v2','zip',dest)
display(FileLink(archive));print('SAMPLE_EXPORT_COMPLETE',archive,flush=True)
