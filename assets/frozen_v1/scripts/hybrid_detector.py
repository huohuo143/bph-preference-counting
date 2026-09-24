"""Native-resolution, locally trained BPH candidates with explicit uncertainty.

No count is copied from another frame. The low-quantile background is used only
to assess contrast, never as a motion requirement or an imputed insect count.
"""
import json,cv2,numpy as np
from functools import lru_cache
import vision as v
import hog_detector as h

VERSION='local-hog-color-20260916-v1'
PARAMS={'hog_hit':.3,'minimum_score':.65,'minimum_gray_q95':.070,
        'minimum_raw_chromatic_median':-.055,'nms_x':48,'nms_y':82}

@lru_cache(maxsize=1)
def model():
    hh=h.hog();hh.setSVMDetector(np.load(v.OUT/'cache/hog_weights.npy'));return hh

def detect(im,group,keep_rejected=False):
    if group=='A':
        ds,q=v.detect(im,group,dict(threshold=.09,min_area=500,min_peak=.17,split_area=5000,min_peak_distance=65))
        for i,d in enumerate(ds):
            reasons=['形态或边缘复核'] if d['suspect'] else []
            if any(i!=j and abs(d['x']-e['x'])<95 and abs(d['y']-e['y'])<125 for j,e in enumerate(ds)):
                reasons.append('相邻或重叠虫体')
            d.update(accepted=True,origin='contrast_color_component',review_reasons=reasons,suspect=bool(reasons),detection_id=i+1)
        return ds,q,ds if keep_rejected else []
    bg=v.background(group);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hp=v.contrast(im)
    contrast=cv2.GaussianBlur(np.minimum(hp-bg['contrast'],hp),(0,0),2.2)
    cr,br=v.chromatic(im)
    boxes,scores=model().detectMultiScale(gray,hitThreshold=PARAMS['hog_hit'],winStride=(8,8),padding=(0,0),scale=1.06,groupThreshold=0)
    candidates=[]
    for i in np.argsort(np.asarray(scores).ravel())[::-1]:
        x,y,w,hh=map(int,boxes[i]);cx=x+w/2;cy=y+hh/2
        if any(((cx-d['x'])/PARAMS['nms_x'])**2+((cy-d['y'])/PARAMS['nms_y'])**2<1 for d in candidates):continue
        sl=np.s_[max(0,round(cy)-36):min(v.H,round(cy)+36),max(0,round(cx)-20):min(v.W,round(cx)+20)]
        g95=float(np.quantile(contrast[sl],.95));raw=float(np.median(cr[sl]));c95=float(np.quantile(cr[sl]-bg['chromatic'][sl],.95))
        dist=float(bg['distance'][min(v.H-1,round(cy)),min(v.W-1,round(cx))])
        reasons=[]
        accepted=bool(scores[i]>=PARAMS['minimum_score'] and g95>=PARAMS['minimum_gray_q95'] and raw>=PARAMS['minimum_raw_chromatic_median'])
        if cy>=v.FLOOR[group]:zone='reflection_or_bottom';reasons.append('水面以下或倒影，未纳入主计数')
        elif dist<=55 and cy<v.FLOOR[group]-20:zone='plant'
        elif cy>=v.FLOOR[group]-80:zone='bottom'
        else:zone='other'
        side=('left' if cx<v.DIVIDER[group] else 'right') if zone=='plant' else ''
        if dist>25 and zone=='plant':reasons.append('植株边缘归属复核')
        if cy>v.FLOOR[group]-120:reasons.append('基部或水面附近')
        if scores[i]<1.1:reasons.append('形态评分较低')
        candidates.append(dict(x=cx,y=cy,bbox=[x,y,w,hh],score=round(float(scores[i]),5),gray_q95=round(g95,5),
            chromatic_median=round(raw,5),chromatic_change_q95=round(c95,5),plant_distance=round(dist,2),
            side=side,zone=zone,accepted=accepted,origin='hog',review_reasons=reasons))
    # Clipped insects at the top edge cannot fit inside a sliding HOG window.
    # A separate native-resolution contrast component detects these candidates.
    strip=np.maximum(contrast[:90],0);mask=(strip>.09).astype(np.uint8)
    n,lab,stats,cents=cv2.connectedComponentsWithStats(mask)
    for j in range(1,n):
        x,y,w,hh,area=map(int,stats[j]);cx,cy=map(float,cents[j])
        if y!=0 or area<400 or w<16 or w>140 or np.max(strip[lab==j])<.15:continue
        if float(np.median(cr[:90][lab==j]))<-.025:continue
        if any(np.hypot(cx-d['x'],cy-d['y'])<100 for d in candidates if d['accepted']):continue
        dist=float(bg['distance'][round(cy),round(cx)])
        if dist>50:continue
        candidates.append(dict(x=cx,y=cy,bbox=[x,y,w,hh],score=None,gray_q95=None,chromatic_median=None,
            chromatic_change_q95=None,plant_distance=round(dist,2),side='left' if cx<v.DIVIDER[group] else 'right',
            zone='plant',accepted=True,origin='top_edge_contrast',review_reasons=['顶部截断虫体']))
    selected=[d for d in candidates if d['accepted']]
    component_params={'A':dict(threshold=.09,min_area=500,min_peak=.17,split_area=5000,min_peak_distance=65),
                      'B':dict(threshold=.075,min_area=500,min_peak=.17,split_area=5000,min_peak_distance=65)}
    if group in component_params:
        blobs,_=v.detect(im,group,component_params[group])
        if group=='A':
            # High-contrast A frames use the calibrated connected-component
            # count. HOG candidates remain an independent review aid.
            selected=[]
            for d in blobs:
                d.update(accepted=True,origin='contrast_color_component',review_reasons=['形态或边缘复核'] if d['suspect'] else [])
                selected.append(d)
        else:
            for d in blobs:
                x,y,w,hh=d['bbox'];cx,cy=round(d['x']),round(d['y'])
                if any(abs(d['x']-e['x'])<65 and abs(d['y']-e['y'])<100 for e in selected):continue
                sl=np.s_[max(0,cy-30):min(v.H,cy+30),max(0,cx-15):min(v.W,cx+15)]
                if not (20<=w<=135 and 35<=hh<=230 and 700<=d['area']<=7500 and d['peak']>=.22):continue
                if float(np.median(cr[sl]))<-.055:continue
                d.update(accepted=True,origin='contrast_color_supplement',review_reasons=['形态检测遗漏的对比度候选'])
                selected.append(d)
    for i,d in enumerate(selected):
        if any(i!=j and abs(d['x']-e['x'])<95 and abs(d['y']-e['y'])<125 for j,e in enumerate(selected)):
            d['review_reasons'].append('相邻或重叠虫体')
        d['suspect']=bool(d['review_reasons'])
    selected.sort(key=lambda d:(d['x'],d['y']))
    for i,d in enumerate(selected,1):d['detection_id']=i
    q=dict(brightness=float(np.median(gray[::8,::8])),sharpness=float(cv2.Laplacian(gray[::4,::4],cv2.CV_32F).var()))
    return selected,q,candidates if keep_rejected else []

def main():
    import argparse,time
    p=argparse.ArgumentParser();p.add_argument('--all-calibration',action='store_true');a=p.parse_args()
    rows=[]
    for g in 'ABC':
        for j in (range(1,21) if a.all_calibration else [1,11,20]):
            sid=f'{g}_cal_{j:02d}';im,_=v.read(v.OUT/'cache/samples'/f'{sid}.JPG');start=time.monotonic();ds,q,reject=detect(im,g)
            row=dict(sample_id=sid,left=sum(d['side']=='left' for d in ds),right=sum(d['side']=='right' for d in ds),detections=ds,**q)
            rows.append(row);cv2.imwrite(str(v.OUT/'qa'/f'{sid}_v1.jpg'),v.overlay(im,g,ds))
            print(sid,row['left'],row['right'],round(time.monotonic()-start,2),flush=True)
    (v.OUT/'data/hybrid_v1_calibration.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
