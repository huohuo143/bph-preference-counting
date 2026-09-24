"""Read-only source inventory and deterministic calibration/holdout sampling."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json, time, hashlib
import numpy as np
import pandas as pd
from PIL import Image

OUT=Path(__file__).resolve().parents[1]

def metadata(r):
    r=dict(r)
    try:
        with Image.open(r['source_path']) as im:
            e=im.getexif(); ex=e.get_ifd(34665) if 34665 in e else {}
            r.update(width=im.width,height=im.height,orientation=e.get(274,1),
                     timestamp=datetime.strptime(ex.get(36867,e.get(306)), '%Y:%m:%d %H:%M:%S').isoformat(),metadata_error='')
    except Exception as exc:r.update(timestamp='',metadata_error=str(exc))
    return r

def select_samples(rows):
    selected=[]
    for g in 'ABC':
        rs=[r for r in rows if r['group']==g]; n=len(rs)
        cal=np.round(np.linspace(0,n-1,20)).astype(int).tolist()
        val=[]
        # One held-out frame in each temporal thirtieth, avoiding calibration
        # and its immediately adjacent one-minute neighbourhood.
        for j in range(30):
            k=int((j+.43)*n/30)
            while any(abs(k-c)<=6 for c in cal+val):k+=7
            val.append(min(k,n-1))
        for split,idx in [('calibration',cal),('holdout',val)]:
            for j,k in enumerate(idx):
                r=dict(rs[k]);r.update(split=split,sample_id=f'{g}_{"cal" if split=="calibration" else "val"}_{j+1:02d}')
                selected.append(r)
    (OUT/'data/sample_selection.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
    pd.DataFrame(selected).to_csv(OUT/'data/校准与留出样本.csv',index=False,encoding='utf-8-sig')
    return selected

def main():
    rows=json.loads((OUT/'data/source_inventory_initial.json').read_text())
    select_samples(rows)
    cache=OUT/'data/metadata_checkpoint.jsonl'
    done={}
    if cache.exists():
        for s in cache.read_text().splitlines():
            r=json.loads(s);done[r['frame_id']]=r
    todo=[r for r in rows if r['frame_id'] not in done]
    start=time.monotonic()
    with cache.open('a') as f,ThreadPoolExecutor(max_workers=8) as pool:
        for i,r in enumerate(pool.map(metadata,todo)):
            f.write(json.dumps(r,ensure_ascii=False)+'\n');done[r['frame_id']]=r
            if (i+1)%1000==0:f.flush();print('metadata',len(done),'/',len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    df=pd.DataFrame([done[r['frame_id']] for r in rows])
    df['timestamp']=pd.to_datetime(df.timestamp,errors='coerce')
    df=df.sort_values(['group','timestamp','relative_path'])
    df['interval_s']=df.groupby('group').timestamp.diff().dt.total_seconds()
    df['elapsed_s']=(df.timestamp-df.groupby('group').timestamp.transform('min')).dt.total_seconds()
    df['timestamp_duplicate']=df.duplicated(['group','timestamp'],keep=False)&df.timestamp.notna()
    df.to_csv(OUT/'data/图片清单.csv',index=False,encoding='utf-8-sig')
    gaps=df[df.interval_s>15].copy()
    gaps.to_csv(OUT/'data/拍摄时间缺口.csv',index=False,encoding='utf-8-sig')
    report=[]
    for g,a in df.groupby('group'):
        report.append(dict(group=g,n=len(a),first=str(a.timestamp.min()),last=str(a.timestamp.max()),
                           intervals={str(k):int(v) for k,v in a.interval_s.value_counts().items()},
                           elapsed_hours=float(a.elapsed_s.max()/3600),
                           potentially_observed_hours=float(a.loc[a.interval_s.between(0,15,inclusive='right'),'interval_s'].sum()/3600),
                           timestamp_duplicates=int(a.timestamp_duplicate.sum()),metadata_errors=int(a.timestamp.isna().sum())))
    (OUT/'data/inventory_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
