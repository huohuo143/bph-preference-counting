"""Time-weighted occupancy; never bridge acquisition or quality gaps."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parents[1]
EPS=1e-8

def make_intervals(times,values,valid,max_gap=15):
    out=[];cumulative=0.;previous=None;segment=-1
    for i in range(len(times)-1):
        a,b=map(float,times[i:i+2]);dt=b-a
        if not(valid[i] and valid[i+1] and np.isfinite(values[i:i+2]).all() and 0<dt<=max_gap):continue
        if previous is None or abs(a-previous)>EPS:segment+=1
        out.append(dict(start=a,end=b,effective_start=cumulative,effective_end=cumulative+dt,
                        va=values[i].copy(),vb=values[i+1].copy(),i=i,j=i+1,segment=segment))
        previous=b;cumulative+=dt
    return out

def rebin(intervals,step,axis='effective',span=None):
    if not intervals:return []
    sk,ek=('effective_start','effective_end') if axis=='effective' else ('start','end')
    span=float(intervals[-1][ek] if span is None else span)
    starts=np.arange(0,span-EPS,step);ends=np.minimum(starts+step,span)
    sums=np.zeros((len(starts),2));dur=np.zeros(len(starts));pieces=np.zeros(len(starts),int)
    for r in intervals:
        a,b=r[sk],r[ek]
        for k in range(int(a//step),min(len(starts),int((b-EPS)//step)+1)):
            lo,hi=max(a,starts[k]),min(b,ends[k])
            if hi<=lo:continue
            vl=r['va']+(r['vb']-r['va'])*((lo-a)/(b-a))
            vh=r['va']+(r['vb']-r['va'])*((hi-a)/(b-a))
            sums[k]+=(vl+vh)/2*(hi-lo);dur[k]+=hi-lo;pieces[k]+=1
    rows=[]
    for k,(a,b) in enumerate(zip(starts,ends)):
        tot=sums[k].sum()
        rows.append(dict(axis=axis,start_s=float(a),end_s=float(b),start_hour=float(a/3600),end_hour=float(b/3600),
                         observed_seconds=float(dur[k]),coverage=float(dur[k]/(b-a)),interval_pieces=int(pieces[k]),
                         left_mean=float(sums[k,0]/dur[k]) if dur[k] else np.nan,
                         right_mean=float(sums[k,1]/dur[k]) if dur[k] else np.nan,
                         left_insect_minutes=float(sums[k,0]/60) if dur[k] else np.nan,right_insect_minutes=float(sums[k,1]/60) if dur[k] else np.nan,
                         left_share=float(sums[k,0]/tot) if tot else np.nan,right_share=float(sums[k,1]/tot) if tot else np.nan))
    direct=sum((r['va']+r['vb'])/2*(r[ek]-r[sk]) for r in intervals)
    assert np.allclose(sums.sum(axis=0),direct,atol=1e-5)
    return rows

def summarize(df):
    bins=[];maps=[];overall=[];gaps=[]
    for group,a in df.groupby('group',sort=True):
        a=a.sort_values(['timestamp','frame_id']).reset_index(drop=True)
        t=a.elapsed_s.to_numpy(float);v=a[['left_count','right_count']].to_numpy(float)
        valid=a.valid_primary.to_numpy(bool)
        intervals=make_intervals(t,v,valid)
        total_seconds=sum(r['end']-r['start'] for r in intervals)
        for axis in ['effective','actual']:
            for step in [300,3600]:
                for row in rebin(intervals,step,axis,span=None if axis=='effective' else t[-1]):
                    bins.append(dict(group=group,group_label=a.group_label.iloc[0],window_seconds=step,**row))
        for r in intervals:
            item=dict(group=group,segment=r['segment']+1,actual_start_s=r['start'],actual_end_s=r['end'],effective_start_s=r['effective_start'],effective_end_s=r['effective_end'],first_frame=a.frame_id.iloc[r['i']],last_frame=a.frame_id.iloc[r['j']])
            if maps and maps[-1]['group']==group and maps[-1]['segment']==item['segment']:
                maps[-1].update(actual_end_s=item['actual_end_s'],effective_end_s=item['effective_end_s'],last_frame=item['last_frame'])
            else:maps.append(item)
        for k in range(1,len(a)):
            if t[k]-t[k-1]>15 or not(valid[k-1] and valid[k]):
                gaps.append(dict(group=group,before=a.frame_id.iloc[k-1],after=a.frame_id.iloc[k],start_s=t[k-1],end_s=t[k],seconds=t[k]-t[k-1],reason='acquisition_gap' if t[k]-t[k-1]>15 else 'quality_gap'))
        integral=sum(((r['va']+r['vb'])/2*(r['end']-r['start']) for r in intervals),start=np.zeros(2))
        denom=float(integral.sum())
        overall.append(dict(group=group,group_label=a.group_label.iloc[0],n_images=len(a),n_valid=int(valid.sum()),n_invalid=int((~valid).sum()),
                            first_timestamp=str(a.timestamp.iloc[0]),last_timestamp=str(a.timestamp.iloc[-1]),actual_hours=float(t[-1]/3600),effective_hours=float(total_seconds/3600),
                            coverage=float(total_seconds/t[-1]) if t[-1] else np.nan,
                            left_mean=float(integral[0]/total_seconds) if total_seconds else np.nan,right_mean=float(integral[1]/total_seconds) if total_seconds else np.nan,
                            left_share=float(integral[0]/denom) if denom else np.nan,right_share=float(integral[1]/denom) if denom else np.nan,
                            left_insect_minutes=float(integral[0]/60) if total_seconds else np.nan,right_insect_minutes=float(integral[1]/60) if total_seconds else np.nan))
    return pd.DataFrame(bins),pd.DataFrame(maps),pd.DataFrame(overall),pd.DataFrame(gaps)

def main():
    df=pd.read_csv(OUT/'data/逐图计数与质量.csv');df['timestamp']=pd.to_datetime(df.timestamp)
    bins,maps,overall,gaps=summarize(df)
    for name,table in [('全部时间窗统计',bins),('拼接时间与原始时间对应',maps),('总体汇总',overall),('分析时间缺口',gaps)]:table.to_csv(OUT/'data'/f'{name}.csv',index=False,encoding='utf-8-sig')
    for axis,label in [('effective','有效时间'),('actual','真实时间')]:
        for step,name in [(300,'每5分钟统计'),(3600,'逐小时统计')]:
            bins[(bins.axis==axis)&(bins.window_seconds==step)].to_csv(OUT/'data'/f'{name}_{label}.csv',index=False,encoding='utf-8-sig')
    print(overall.to_string(index=False))

if __name__=='__main__':main()
