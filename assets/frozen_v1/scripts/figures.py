"""Chinese scientific figures from audited, time-weighted count tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

OUT=Path(__file__).resolve().parents[1]
COLORS={'left':'#2677A8','right':'#D86C32'}
LABELS={'A':'ZH11 B3-OE','B':'BPH33','C':'BPH33回补'}
plt.rcParams.update({'font.family':'Arial Unicode MS','font.size':10,
 'axes.linewidth':.8,'axes.spines.top':False,'axes.spines.right':False,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})

def save(fig,name):
    (OUT/'figures').mkdir(exist_ok=True)
    for ext in ['svg','pdf','png']:
        fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def marker(ax,x,label='衔接'):
    trans=blended_transform_factory(ax.transData,ax.transAxes)
    ax.plot([x,x],[-.025,.025],color='#6B667B',lw=1.4,transform=trans,clip_on=False,zorder=8)
    ax.annotate(label,xy=(x,0),xycoords=trans,xytext=(0,-25),textcoords='offset points',
                ha='center',va='top',fontsize=8,color='#6B667B',clip_on=False)

def curves(bins,mapping,overall,actual=False,provisional=False):
    fig,axes=plt.subplots(3,1,figsize=(12,9.6),layout='constrained')
    fig.set_constrained_layout_pads(h_pad=.16,hspace=.18)
    axis='actual' if actual else 'effective'
    for ax,g in zip(axes,'ABC'):
        a=bins[(bins.group==g)&(bins.axis==axis)&(bins.window_seconds==300)]
        o=overall[overall.group==g].iloc[0]
        if len(a):
            x=(a.start_hour+a.end_hour)/2
            for side,lab in [('left','左侧对照'),('right','右侧抗性')]:
                # Actual-time bins without observation remain NaN. The effective
                # plot deliberately joins observation segments per user request.
                y=a[f'{side}_mean'].to_numpy(float)
                if actual:
                    # Break every acquisition/quality gap, including a short
                    # gap entirely inside one five-minute bin. The segmented
                    # real-time table retains the same bin boundaries.
                    for _,part in a.groupby('segment',sort=True) if 'segment' in a else [(0,a)]:
                        xx=(part.start_hour+part.end_hour)/2
                        ax.plot(xx,part[f'{side}_mean'],color=COLORS[side],lw=1.2,label=lab)
                else:
                    ax.plot(x,y,color=COLORS[side],lw=1.2,label=lab)
            ymax=np.nanmax(a[['left_mean','right_mean']].to_numpy())
            ax.set_ylim(0,max(5,ymax*1.15))
        else:
            ax.text(.5,.5,'计数验收未通过，主统计留空' if not provisional else '尚无可用的连续计数区间',ha='center',va='center',transform=ax.transAxes)
            ax.set_ylim(0,1)
        span=float(o.actual_hours if actual else o.effective_hours)
        ax.set_xlim(0,max(span,.1))
        if not len(a):ax.set_xticks([]);ax.set_yticks([])
        ax.set_ylabel('株上候选虫数（只）' if provisional else '株上可见虫数（只）')
        ax.set_title(f'{g}  {LABELS[g]}',loc='left',fontweight='bold',pad=10)
        extra='  |  未通过计数验收' if provisional and g!='A' else '  |  模型目视自检达标' if g=='A' else ''
        ax.text(1,1.035,f'照片 {int(o.n_images):,} 张  |  纳入时长 {o.effective_hours:.2f} 小时'+extra,ha='right',transform=ax.transAxes,fontsize=8,color='#666666')
        ax.grid(axis='y',color='#E6E6E6',lw=.5)
        ax.set_axisbelow(True)
        if not actual and len(mapping):
            points=mapping[mapping.group==g].iloc[1:].effective_start_s.to_numpy(float)/3600
            # All splice locations are retained in the correspondence table.
            # Coincident ticks share a single label to keep the axis readable.
            last=-1e20
            for point in points:
                marker(ax,point,'衔接' if point-last>max(span*.035,.3) else '')
                if point-last>max(span*.035,.3):last=point
        ax.set_xlabel('真实经过时间（小时）' if actual else '累计有效观察时间（小时）',labelpad=22 if not actual else 8)
    handles=[Line2D([0],[0],color=COLORS[s],lw=2,label=l) for s,l in [('left','左侧对照'),('right','右侧抗性')]]
    axes[0].legend(handles=handles,loc='upper right',frameon=False,ncol=2)
    title='褐飞虱株上可见数量随时间变化'
    subtitle='每5分钟的时间加权均值；曲线衔接不补计停拍时长' if not actual else '每5分钟的时间加权均值；横轴保留真实停拍时长'
    if provisional:title+='（自动候选计数，待复核）'
    fig.suptitle(title+'\n'+subtitle,fontsize=13)
    save(fig,('03_真实时间曲线_保留缺口_中文' if actual else '01_时间曲线_有效时间拼接_中文')+('_待复核' if provisional else ''))

def heatmap(bins,mapping,overall,provisional=False):
    fig,axes=plt.subplots(3,1,figsize=(13,7),layout='constrained')
    fig.set_constrained_layout_pads(h_pad=.15,hspace=.3)
    cmap=LinearSegmentedColormap.from_list('occupancy',['#FFF8CC','#FFD77A','#F4A34F','#C96730','#7B381E'])
    cmap.set_bad('#E6E6E6')
    mesh=None
    for ax,g in zip(axes,'ABC'):
        a=bins[(bins.group==g)&(bins.axis=='effective')&(bins.window_seconds==3600)].sort_values('start_s')
        if not len(a):
            ax.text(.5,.5,'计数验收未通过，主统计留空',ha='center',va='center',transform=ax.transAxes)
            ax.set_title(f'{g}  {LABELS[g]}',loc='left',fontweight='bold')
            ax.set_yticks([.5,1.5],['左侧对照','右侧抗性']);ax.set_ylim(2,0)
            ax.set_xticks([])
            continue
        z=a[['left_share','right_share']].to_numpy(float).T*100
        edges=np.r_[a.start_hour.to_numpy(),a.end_hour.iloc[-1]]
        mesh=ax.pcolormesh(edges,[0,1,2],z,cmap=cmap,vmin=0,vmax=100,edgecolors='white',linewidth=.65)
        for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            for i in range(2):
                if np.isfinite(z[i,j]) and hi-lo>.3:
                    ax.text((lo+hi)/2,i+.5,f'{z[i,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if z[i,j]>70 else '#49361F')
        ax.set_ylim(2,0);ax.set_yticks([.5,1.5],['左侧对照','右侧抗性'])
        ax.set_xlim(0,edges[-1]);ax.set_xticks(np.arange(0,edges[-1]+.01,2))
        ax.set_title(f'{g}  {LABELS[g]}',loc='left',fontweight='bold')
        ax.set_xlabel('累计有效观察时间（小时）',labelpad=20)
        last=-1e20
        for p in mapping[mapping.group==g].iloc[1:].effective_start_s.to_numpy(float)/3600:
            marker(ax,p,'衔接' if p-last>max(edges[-1]*.035,.3) else '')
            if p-last>max(edges[-1]*.035,.3):last=p
    if mesh is not None:
        cb=fig.colorbar(mesh,ax=axes,location='right',shrink=.88,pad=.025)
        cb.set_label('株上累计占据量占比（%）');cb.set_ticks([0,20,40,60,80,100])
    title='褐飞虱逐小时停留分布'
    if provisional:title+='（自动候选计数，待复核）'
    fig.suptitle(title+'\n左右两侧“虫数×时间”积分的比例；灰色表示无法计算',fontsize=13)
    save(fig,'02_逐小时分布_有效时间拼接_中文'+('_待复核' if provisional else ''))

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--provisional',action='store_true');a=p.parse_args()
    bins=pd.read_csv(OUT/'data/全部时间窗统计.csv')
    mapping=pd.read_csv(OUT/'data/拼接时间与原始时间对应.csv')
    overall=pd.read_csv(OUT/'data/总体汇总.csv')
    curves(bins,mapping,overall,provisional=a.provisional)
    curves(bins,mapping,overall,actual=True,provisional=a.provisional)
    heatmap(bins,mapping,overall,provisional=a.provisional)
    print(OUT/'figures')

if __name__=='__main__':main()
