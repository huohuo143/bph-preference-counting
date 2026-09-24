"""Export audited records; failed count validation never becomes primary data.

Run only after the native-resolution full batch finishes. Candidate summaries
are explicitly exploratory, and are separate from the primary-count tables.
"""
from pathlib import Path
import json,csv,gzip,hashlib,collections,sqlite3
import numpy as np
import pandas as pd
from summarize import summarize,make_intervals,rebin
import figures

OUT=Path(__file__).resolve().parents[1]
RUN=OUT/'data/run_24cc7e7f926b5b7e'
LABELS={'A':'ZH11 B3-OE','B':'BPH33','C':'BPH33回补'}

def save_csv(df,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False,encoding='utf-8-sig',na_rep='')

def validation():
    refs=json.loads((OUT/'data/independent_visual_reference.json').read_text())
    comparisons=[];metrics=[]
    for version,run in [('v1','4e84dd351c3bac11'),('v2_候选修正未采用','0974eb1d99dfc8f5')]:
        preds={r['sample_id']:r for r in map(json.loads,(OUT/f'data/run_{run}/samples_checkpoint.jsonl').open())}
        for r in refs['rows']:
            pred=preds[r['sample_id']]
            for side in ['left','right']:
                target=r[side+'_count'];actual=pred['auto_'+side+'_count']
                comparisons.append(dict(version=version,group=r['sample_id'][0],sample_id=r['sample_id'],side=side,
                    reference_count=target,auto_count=actual,error=actual-target if target is not None else None,
                    reference_status=r['confidence'],notes=r['notes'],reviewer=refs['reviewer'],
                    inspection_before_predictions=True,independent_human_expert=False))
    comparison=pd.DataFrame(comparisons)
    for (version,g,side),a in comparison.groupby(['version','group','side'],sort=False):
        e=a.error.dropna().to_numpy(float)
        mae=float(np.mean(np.abs(e))) if len(e) else np.nan
        within=float(np.mean(np.abs(e)<=2)) if len(e) else np.nan
        numerical=bool(len(e)>0 and mae<=1 and within>=.95)
        # All 30 prespecified samples must have usable reference counts.
        passed=bool(len(e)==30 and numerical and version=='v1')
        metrics.append(dict(version=version,group=g,group_label=LABELS[g],side=side,
            selected_samples=len(a),usable_reference_sides=len(e),unresolved_reference_sides=len(a)-len(e),
            mae=mae,within_two_fraction=within,numerical_target_on_usable_sides=numerical,
            initial_model_visual_check_passed=passed,independent_human_expert=False,
            evaluation_role='初始留出核验' if version=='v1' else '首轮失败后探索性对照；不可作为新的留出验收'))
    save_csv(comparison,OUT/'data/逐样本核验对照.csv')
    save_csv(pd.DataFrame(metrics),OUT/'data/计数核验汇总.csv')
    return refs,comparison,pd.DataFrame(metrics)

def export_frames(refs,passed_groups):
    inventory=pd.read_csv(OUT/'data/图片清单.csv')
    inventory['timestamp']=pd.to_datetime(inventory.timestamp)
    inv=inventory.set_index('frame_id').to_dict('index')
    samples=json.loads((OUT/'data/sample_selection.json').read_text())
    samplemap={r['frame_id']:r['sample_id'] for r in samples}
    refmap={r['sample_id']:r for r in refs['rows']}
    rows=[];seen=set();hashmap=collections.defaultdict(list);coord_n=0
    db=sqlite3.connect(OUT/'data/自动检测索引.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS frames(frame_id TEXT PRIMARY KEY, payload TEXT)')
    coord_path=OUT/'data/全部自动检测坐标.csv.gz'
    coord_fields=['frame_id','group','source_path','detector_fingerprint','detection_id','x','y','bbox_x','bbox_y','bbox_width','bbox_height','side','zone','score','origin','suspect','review_reasons']
    with gzip.open(coord_path,'wt',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,coord_fields);writer.writeheader()
        for line in (RUN/'full_checkpoint.jsonl').open():
            r=json.loads(line);fid=r['frame_id']
            if fid in seen:raise ValueError('duplicate processed frame '+fid)
            seen.add(fid);base=inv[fid];g=r['group'];sid=samplemap.get(fid);ref=refmap.get(sid)
            reason=list(r.get('quality_reasons',[]))
            # C_val_29 demonstrates defocus missed by a universal threshold.
            # This is an explicit review flag, not a newly validated classifier.
            if g=='C' and r.get('sharpness',0)<90:reason.append('回补组清晰度偏低，候选统计暂不纳入；阈值待复核')
            image_usable=r['status']=='processed' and not reason
            group_passed=g in passed_groups
            reliable_auto=bool(image_usable and group_passed)
            # Candidate plots describe the algorithm output consistently across
            # all frames. They do not selectively remove only visually sampled
            # overlapping insects while retaining unreviewed similar frames.
            candidate_usable=image_usable
            row=dict(frame_id=fid,**base,sample_id=sid,processing_status=r['status'],
                processed_at=r.get('processed_at'),source_sha256=r.get('sha256'),source_unchanged=r.get('source_unchanged'),
                detector_version='local-hog-color-20260916-v1',detector_fingerprint='24cc7e7f926b5b7e',
                quality_ok=image_usable,quality_reason='；'.join(reason),brightness=r.get('brightness'),sharpness=r.get('sharpness'),
                scene_overlap=r.get('scene_overlap'),auto_left_count=r.get('auto_left_count'),auto_right_count=r.get('auto_right_count'),
                auto_other_count=r.get('auto_other_count'),reflection_or_bottom_candidates=r.get('reflection_or_bottom_count'),
                suspect_count=r.get('suspect_count'),left_count=r.get('auto_left_count') if reliable_auto else None,
                right_count=r.get('auto_right_count') if reliable_auto else None,other_count=None,
                other_missing_reason='其他位置总数未完成可靠核验，仅提供自动候选',
                valid_primary=reliable_auto,valid_candidate_summary=candidate_usable,
                primary_missing_reason='' if reliable_auto else '该组未通过初始计数验收' if not group_passed else '图片质量不能可靠计数',
                review_status='模型目视参考核验；非人工专家' if ref else '自动逐图处理；尚未逐图目视复核',
                reference_left_count=ref['left_count'] if ref else None,reference_right_count=ref['right_count'] if ref else None,
                reference_notes=ref['notes'] if ref else '',expert_reviewed=False,
                review_required=True,annotation_path=r.get('annotation_path',''),error=r.get('error'))
            rows.append(row)
            db.execute('INSERT OR REPLACE INTO frames VALUES(?,?)',(fid,json.dumps(r,ensure_ascii=False)))
            if len(rows)%1000==0:db.commit()
            if r.get('sha256'):hashmap[r['sha256']].append(fid)
            ds=r.get('detections',[])
            if r['status']=='processed':
                for side in ['left','right']:
                    assert sum(d.get('side')==side for d in ds)==r['auto_'+side+'_count'],fid
            for d in ds:
                x,y,w,h=d['bbox']
                assert 0<=x<4416 and 0<=y<2488 and w>0 and h>0 and x+w<=4416 and y+h<=2488,(fid,d)
                writer.writerow(dict(frame_id=fid,group=g,source_path=base['source_path'],detector_fingerprint='24cc7e7f926b5b7e',
                    detection_id=d['detection_id'],x=d['x'],y=d['y'],bbox_x=x,bbox_y=y,bbox_width=w,bbox_height=h,
                    side=d.get('side'),zone=d.get('zone'),score=d.get('score'),origin=d.get('origin'),suspect=d.get('suspect'),
                    review_reasons='；'.join(d.get('review_reasons',[]))))
                coord_n+=1
    db.commit();db.close()
    assert seen==set(inv),f'Incomplete processing: {len(seen)} of {len(inv)}'
    df=pd.DataFrame(rows).sort_values(['group','timestamp','frame_id']).reset_index(drop=True)
    for _,idx in df.groupby('group',sort=True).groups.items():
        a=df.loc[idx];dt=a.elapsed_s.diff().to_numpy(float)
        for valid_col,outcol in [('quality_ok','image_effective_elapsed_s'),('valid_primary','primary_effective_elapsed_s'),('valid_candidate_summary','candidate_effective_elapsed_s')]:
            valid=a[valid_col].to_numpy(bool);good=valid&np.r_[False,valid[:-1]]&(dt>0)&(dt<=15)
            df.loc[idx,outcol]=np.cumsum(np.where(good,dt,0))
    save_csv(df,OUT/'data/逐图计数与质量.csv')
    save_csv(df[['frame_id','group','source_path','source_sha256','source_unchanged']],OUT/'data/源文件哈希与完整性.csv')
    review_columns=['frame_id','group_label','source_path','timestamp','auto_left_count','auto_right_count','auto_other_count','suspect_count','quality_reason','primary_missing_reason','review_status','reference_left_count','reference_right_count','reference_notes','annotation_path']
    save_csv(df[review_columns],OUT/'data/异常与复核清单.csv')
    duplicates=[dict(sha256=k,n_files=len(v),frame_ids=';'.join(v)) for k,v in hashmap.items() if len(v)>1]
    save_csv(pd.DataFrame(duplicates,columns=['sha256','n_files','frame_ids']),OUT/'data/完全相同文件检查.csv')
    audit=dict(total_inventory=len(inventory),total_processed=len(df),missing_records=0,duplicate_frame_ids=0,
        processing_errors=int((df.processing_status!='processed').sum()),source_unchanged_count=int(df.source_unchanged.sum()),
        coordinate_count=coord_n,coordinate_bounds_passed=True,automatic_counts_match_coordinates=True,
        byte_identical_file_groups=len(duplicates),group_images=df.groupby('group').size().to_dict())
    (OUT/'qa/全量记录验收.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    return inventory,df,audit

def tables_and_figures(df,candidate=False):
    folder=OUT/'data'/('候选统计_未通过验收' if candidate else '主统计_仅初检达标范围')
    folder.mkdir(exist_ok=True)
    a=df.copy()
    if candidate:
        a['left_count']=a.auto_left_count;a['right_count']=a.auto_right_count
        a['valid_primary']=a.valid_candidate_summary
    bins,mapping,overall,gaps=summarize(a)
    for table in [bins,mapping,overall,gaps]:table['data_status']='自动候选，待复核' if candidate else '仅模型目视初检达标范围；非专家逐图确认'
    for name,table in [('全部时间窗统计',bins),('拼接时间与原始时间对应',mapping),('总体汇总',overall),('分析时间缺口',gaps)]:save_csv(table,folder/f'{name}.csv')
    for axis,label in [('effective','有效时间'),('actual','真实时间')]:
        for step,name in [(300,'每5分钟统计'),(3600,'逐小时统计')]:save_csv(bins[(bins.axis==axis)&(bins.window_seconds==step)],folder/f'{name}_{label}.csv')
    # Segment-specific five-minute values for the real-time figure. Split bins
    # contain only the observations within the corresponding segment.
    actual=[]
    for g,part in a.groupby('group'):
        part=part.sort_values(['timestamp','frame_id'])
        intervals=make_intervals(part.elapsed_s.to_numpy(float),part[['left_count','right_count']].to_numpy(float),part.valid_primary.to_numpy(bool))
        byseg=collections.defaultdict(list)
        for r in intervals:byseg[r['segment']].append(r)
        for segment,items in byseg.items():
            for r in rebin(items,300,'actual',span=part.elapsed_s.iloc[-1]):
                if r['observed_seconds']<=0:continue
                r['start_s']=max(r['start_s'],items[0]['start']);r['end_s']=min(r['end_s'],items[-1]['end'])
                r['start_hour']=r['start_s']/3600;r['end_hour']=r['end_s']/3600
                actual.append(dict(group=g,group_label=LABELS[g],window_seconds=300,segment=segment+1,**r))
    actual=pd.DataFrame(actual)
    save_csv(actual,folder/'真实时间分段绘图数据.csv')
    figures.curves(bins,mapping,overall,provisional=candidate)
    figures.curves(actual,mapping,overall,actual=True,provisional=candidate)
    figures.heatmap(bins,mapping,overall,provisional=candidate)
    return folder,bins,mapping,overall,gaps

def main():
    refs,comparison,metrics=validation()
    first=metrics[metrics.version=='v1']
    passed_groups={g for g,a in first.groupby('group') if len(a)==2 and a.initial_model_visual_check_passed.all()}
    inventory,df,audit=export_frames(refs,passed_groups)
    audit['scope']='处理记录、坐标与导出一致性；不等同于全部计数精度验收'
    audit['initial_model_visual_check_passed_groups']=sorted(passed_groups)
    audit['all_groups_count_acceptance_completed']=passed_groups==set(LABELS)
    audit['independent_human_expert_acceptance_completed']=False
    from priority_evidence import build_evidence
    audit['priority_annotated_frames']=len(build_evidence(df))
    primary=tables_and_figures(df)
    candidate=tables_and_figures(df,True)
    from report_results import deliver
    deliver(inventory,df,audit,comparison,metrics,primary,candidate)
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    print('输出目录：',OUT)

if __name__=='__main__':main()
