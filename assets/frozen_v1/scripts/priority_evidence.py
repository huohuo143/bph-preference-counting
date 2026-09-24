"""Prioritize review using transparent flags; never alter counts."""
from pathlib import Path
import json,sqlite3,collections
import cv2
import pandas as pd
import vision

OUT=Path(__file__).resolve().parents[1]

def build_evidence(df):
    reasons=collections.defaultdict(set)
    def add(part,reason):
        for fid in part.frame_id:reasons[fid].add(reason)
    for g,a in df.groupby('group'):
        a=a.sort_values('timestamp').copy()
        add(a.nsmallest(3,'sharpness'),'本组清晰度最低候选')
        add(a.nlargest(3,'suspect_count'),'本组疑似目标数较多')
        a['count_change']=a[['auto_left_count','auto_right_count']].diff().abs().max(axis=1)
        add(a[(a.interval_s>0)&(a.interval_s<=15)].nlargest(3,'count_change'),'相邻照片计数变化较大，可能是真实移动或计数问题')
        for k in range(1,len(a)):
            if a.interval_s.iloc[k]>15:
                add(a.iloc[k-1:k+1],'停拍缺口前后')
    for _,r in df[df.reference_left_count.notna()|df.reference_right_count.notna()].iterrows():
        if ((pd.notna(r.reference_left_count) and abs(r.auto_left_count-r.reference_left_count)>2)
            or (pd.notna(r.reference_right_count) and abs(r.auto_right_count-r.reference_right_count)>2)):
            reasons[r.frame_id].add('初始核验某侧差异超过2只')
    dest=OUT/'evidence/优先复核';dest.mkdir(exist_ok=True)
    rows=[]
    with sqlite3.connect(f'file:{OUT}/data/自动检测索引.sqlite?mode=ro',uri=True) as db:
        for fid,flags in reasons.items():
            row=df[df.frame_id==fid].iloc[0]
            payload=json.loads(db.execute('SELECT payload FROM frames WHERE frame_id=?',(fid,)).fetchone()[0])
            im,_=vision.read(Path(row.source_path))
            path=dest/f'{fid}_自动候选标注.jpg'
            ok=cv2.imwrite(str(path),vision.overlay(im,row.group,payload['detections']))
            if not ok:raise OSError(path)
            rows.append(dict(frame_id=fid,group=row.group,source_path=row.source_path,timestamp=row.timestamp,
                reason='；'.join(sorted(flags)),annotation_path=str(path),auto_left_count=row.auto_left_count,
                auto_right_count=row.auto_right_count,sharpness=row.sharpness,suspect_count=row.suspect_count,
                expert_reviewed=False))
    result=pd.DataFrame(rows).sort_values(['group','timestamp'])
    result.to_csv(OUT/'data/优先复核照片.csv',index=False,encoding='utf-8-sig')
    (dest/'标注说明.md').write_text('''# 优先复核标注图

蓝框为左侧株上候选，橙框为右侧株上候选，紫框为其他或疑似位置；底部紫线仅为场景区域界线。数字对应同一照片的自动检测顺序。

这些是算法候选框，不是已经逐体人工确认的虫数。选择依据包括清晰度、疑似目标数、相邻计数变化、停拍前后及留出核验偏差；异常信号不必然等于计数错误。完整索引见 data/优先复核照片.csv。

预览缩放到2048像素宽便于查看，全部坐标仍用原图4416×2488像素；需要查看细部时请打开原图或本机复核页面。原图未被改动。
''')
    return result

if __name__=='__main__':
    df=pd.read_csv(OUT/'data/逐图计数与质量.csv')
    print('优先复核标注图：',len(build_evidence(df)))
