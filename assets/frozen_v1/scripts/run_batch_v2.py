"""Resumable, source-preserving, one-native-image-per-record batch processing."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,time,hashlib,traceback,platform
from concurrent.futures import ProcessPoolExecutor,as_completed
import cv2,numpy as np,pandas as pd
import vision as v
import hybrid_detector_v2 as detector

OUT=v.OUT
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fingerprint():
    files=[OUT/'scripts'/p for p in ['vision.py','hog_detector.py','hybrid_detector_v2.py']]
    files += [OUT/'cache/hog_weights.npy']+[OUT/'cache'/f'{g}_background.npz' for g in 'ABC']
    hashes={str(p.relative_to(OUT)):sha(p) for p in files}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    return dict(version=detector.VERSION,fingerprint=key,files=hashes,parameters=detector.PARAMS,
                python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__,created_at=pd.Timestamp.now().isoformat())

def process(row):
    start=time.monotonic();item={k:row[k] for k in ['frame_id','group','source_path','sequence_index']}
    item.update(sample_id=row.get('sample_id'),processed_at=pd.Timestamp.now().isoformat(),status='error',error=None)
    try:
        source=Path(row['source_path']);before=source.stat()
        # Cached samples have been hash-checked when copied from the source.
        cached=OUT/'cache/samples'/f"{row.get('sample_id')}.JPG"
        loadpath=cached if row.get('use_cache') and cached.exists() else source
        im,raw=v.read(loadpath)
        item['sha256']=hashlib.sha256(raw).hexdigest()
        item['source_unchanged']=bool(before.st_size==row['size_bytes'] and before.st_mtime_ns==row['mtime_ns'])
        ds,q,candidates=detector.detect(im,row['group'],keep_rejected=True)
        g=row['group'];small=cv2.resize(im,(1104,622));hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
        green=((hsv[:,:,0]>=30)&(hsv[:,:,0]<=95)&(hsv[:,:,1]>=60)).astype(np.uint8)
        green[round(v.FLOOR[g]/4):]=0
        roi=cv2.resize(v.background(g)['plant'],(1104,622),interpolation=cv2.INTER_NEAREST)
        roi=cv2.dilate(roi,np.ones((21,21),np.uint8))>0
        overlap=float((green.astype(bool)&roi).sum()/max(green.sum(),1))
        reasons=[]
        if not item['source_unchanged']:reasons.append('源文件大小或修改时间与清单不一致')
        if not 45<=q['brightness']<=235:reasons.append('曝光异常')
        if q['sharpness']<20:reasons.append('严重失焦')
        if overlap<.35:reasons.append('植株画面位置或颜色显著变化')
        item.update(status='processed',quality_ok=not reasons,quality_reasons=reasons,scene_overlap=overlap,**q,
                    auto_left_count=sum(d['side']=='left' for d in ds),auto_right_count=sum(d['side']=='right' for d in ds),
                    auto_other_count=sum(d['zone'] in ['other','bottom'] for d in ds),
                    reflection_or_bottom_count=sum(d['zone']=='reflection_or_bottom' for d in ds),
                    suspect_count=sum(bool(d.get('suspect')) for d in ds),
                    detections=ds,candidates=candidates)
        if row.get('sample_id'):
            name=f"{row['sample_id']}_v2_计数标注.jpg"
            cv2.imwrite(str(OUT/'evidence'/name),v.overlay(im,g,ds))
            item['annotation_path']=str(OUT/'evidence'/name)
    except Exception as exc:
        item.update(quality_ok=False,quality_reasons=['读取或处理失败'],error=f'{type(exc).__name__}: {exc}',detections=[],candidates=[])
    item['processing_seconds']=round(time.monotonic()-start,4)
    return item

def read_existing(path):
    done={};valid_end=0
    if not path.exists():return done
    with path.open('rb') as f:
        for line in f:
            try:r=json.loads(line);done[r['frame_id']]=r;valid_end=f.tell()
            except json.JSONDecodeError:break
    if valid_end<path.stat().st_size:
        backup=path.with_suffix('.interrupted-tail.jsonl')
        backup.write_bytes(path.read_bytes()[valid_end:])
        with path.open('r+b') as f:f.truncate(valid_end)
    return done

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['samples','full'],default='samples')
    p.add_argument('--workers',type=int,default=12);p.add_argument('--limit',type=int);a=p.parse_args()
    (OUT/'evidence').mkdir(exist_ok=True)
    inventory=pd.read_csv(OUT/'data/图片清单.csv').to_dict('records')
    samples=json.loads((OUT/'data/sample_selection.json').read_text())
    sid={s['frame_id']:s['sample_id'] for s in samples}
    rows=[]
    for r in inventory:
        if a.mode=='samples' and r['frame_id'] not in sid:continue
        r['sample_id']=sid.get(r['frame_id']);r['use_cache']=a.mode=='samples';rows.append(r)
    if a.limit:rows=rows[:a.limit]
    meta=fingerprint();run_dir=OUT/'data'/f"run_{meta['fingerprint']}";run_dir.mkdir(exist_ok=True)
    mp=run_dir/'detector_manifest.json'
    if not mp.exists():mp.write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    path=run_dir/f'{a.mode}_checkpoint.jsonl';done=read_existing(path)
    todo=[r for r in rows if r['frame_id'] not in done]
    print(json.dumps(dict(mode=a.mode,total=len(rows),completed=len(done),remaining=len(todo),run_dir=str(run_dir)),ensure_ascii=False),flush=True)
    status_file=run_dir/f'{a.mode}_progress.json';start=time.monotonic();n=0;errors=sum(r['status']=='error' for r in done.values())
    with path.open('a',buffering=1) as f,ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs={pool.submit(process,r):r['frame_id'] for r in todo}
        for future in as_completed(jobs):
            result=future.result();f.write(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n');n+=1
            if result['status']=='error':errors+=1
            if n%100==0 or n==len(todo):
                progress=dict(mode=a.mode,total=len(rows),completed=len(done)+n,errors=errors,elapsed_seconds=round(time.monotonic()-start,2),
                              images_per_second=round(n/(time.monotonic()-start),3),last_frame=result['frame_id'],updated_at=pd.Timestamp.now().isoformat())
                status_file.write_text(json.dumps(progress,ensure_ascii=False,indent=2));print(json.dumps(progress,ensure_ascii=False),flush=True)
    print('FINISHED',path,flush=True)
    (OUT/'data'/f'latest_{a.mode}_run.json').write_text(json.dumps(dict(run_dir=str(run_dir),checkpoint=str(path),fingerprint=meta['fingerprint']),indent=2))

if __name__=='__main__':main()
