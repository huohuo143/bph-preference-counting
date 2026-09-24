"""Native-resolution BPH candidate detection; no pretrained/external service.

Background suppression uses temporally separated calibration images only.
Per-frame local contrast is always retained; no interpolation of frame counts.
"""
from pathlib import Path
from functools import lru_cache
import json,cv2,numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

OUT=Path(__file__).resolve().parents[1]
cv2.setNumThreads(1)
W,H=4416,2488
FLOOR={'A':2190,'B':2410,'C':2240}
DIVIDER={'A':2100,'B':2050,'C':1980}
PLANT_BOUNDS={'A':[(390,1890),(2240,3820)],'B':[(350,1900),(2240,4020)],'C':[(480,1810),(2180,3520)]}
PARAMS={'threshold':.042,'min_area':110,'min_peak':.080,'smooth':2.2,'split_area':2000,'min_peak_distance':35}

def chromatic(im):
    med=np.median(im[250:1700:3,2000:2100:3].reshape(-1,3),axis=0).astype(np.float32)
    b,g,r=cv2.split(im.astype(np.float32));r=r*(med[1]/med[2]);b=b*(med[1]/med[0])
    return (r-g)/(r+g+10),(r-b)/(r+b+10)

def add_chromatic_background(group):
    arr=[]
    for j in range(1,21):
        im,_=read(OUT/'cache/samples'/f'{group}_cal_{j:02d}.JPG')
        arr.append(chromatic(im)[0].astype(np.float16))
    z=background(group);z['chromatic']=np.quantile(np.stack(arr),.15,axis=0).astype(np.float32)
    np.savez_compressed(OUT/'cache'/f'{group}_background.npz',**z)

def read(path):
    raw=Path(path).read_bytes();im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if im is None:raise ValueError('JPEG decoding failed')
    if im.shape[:2]!=(H,W):raise ValueError(f'unexpected shape {im.shape}')
    return im,raw

def contrast(im):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred=cv2.GaussianBlur(gray,(0,0),25)
    return (blurred-gray)/(blurred+10)

def prepare_background(group):
    arrays=[];colors=[]
    for j in range(1,21):
        im,_=read(OUT/'cache/samples'/f'{group}_cal_{j:02d}.JPG')
        arrays.append(contrast(im).astype(np.float16))
        colors.append(cv2.resize(im,(1104,622),interpolation=cv2.INTER_AREA))
    # Low contrast quantile suppresses recurrent plant edges without assuming
    # that an insect is moving between consecutive frames.
    stack=np.stack(arrays)
    bg=np.quantile(stack,.15,axis=0).astype(np.float32)
    median=np.median(np.stack(colors),axis=0).astype(np.uint8)
    medfull=cv2.resize(median,(W,H))
    hsv=cv2.cvtColor(medfull,cv2.COLOR_BGR2HSV)
    green=((hsv[:,:,0]>=30)&(hsv[:,:,0]<=95)&(hsv[:,:,1]>=60)).astype(np.uint8)*255
    green=cv2.morphologyEx(green,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,51)))
    green[FLOOR[group]:]=0
    n,l,s,c=cv2.connectedComponentsWithStats(green)
    plant=np.zeros_like(green)
    for i in range(1,n):
        if s[i,4]>12000:plant[l==i]=255
    allowed=np.zeros(W,bool)
    for lo,hi in PLANT_BOUNDS[group]:allowed[lo:hi]=True
    plant[:,~allowed]=0
    distance=cv2.distanceTransform(255-plant,cv2.DIST_L2,5)
    np.savez_compressed(OUT/'cache'/f'{group}_background.npz',contrast=bg,plant=plant,distance=distance)
    cv2.imwrite(str(OUT/'qa'/f'{group}_plant_mask.jpg'),cv2.resize(plant,(2048,1154)))
    del stack,arrays

@lru_cache(maxsize=3)
def background(group):
    z=np.load(OUT/'cache'/f'{group}_background.npz')
    return {k:z[k] for k in z.files}

def detect(im,group,params=None):
    p={**PARAMS,**(params or {})};bg=background(group)
    hp=contrast(im)
    residual=cv2.GaussianBlur(np.minimum(hp-bg['contrast'],hp),(0,0),p['smooth'])
    if 'chromatic' in bg:
        score,brown=chromatic(im)
        cr=np.maximum(score-bg['chromatic']-.012,0)*3.5
        cr[(score<-.02)|(brown<.04)]=0
        cr=cv2.GaussianBlur(cr,(0,0),p['smooth'])
        residual=np.maximum(residual,cr)
    mask=(residual>p['threshold']).astype(np.uint8)
    # The part below the substrate/water boundary is dominated by reflections.
    mask[FLOOR[group]+30:]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,labels,stats,cents=cv2.connectedComponentsWithStats(mask,8)
    detections=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if area<p['min_area'] or area>200000 or w>750:continue
        local=(labels[y:y+h,x:x+w]==i).astype(np.uint8)
        response=residual[y:y+h,x:x+w]
        if float(response[local>0].max())<p['min_peak']:continue
        parts=[local]
        if area>p['split_area']:
            score=cv2.GaussianBlur(response,(0,0),7)
            coords=peak_local_max(score,min_distance=p['min_peak_distance'],threshold_abs=p['min_peak'],labels=local,exclude_border=False)
            if 1<len(coords)<=60:
                markers=np.zeros(local.shape,np.int32)
                for j,(cy,cx) in enumerate(coords,1):markers[cy,cx]=j
                split=watershed(-score,markers,mask=local)
                parts=[(split==j).astype(np.uint8) for j in range(1,len(coords)+1)]
        for part in parts:
            yy,xx=np.where(part>0)
            if len(xx)<p['min_area'] or len(xx)>12000:continue
            values=response[yy,xx]
            peak=float(values.max()); ww=values.clip(.001)
            cx=float(np.average(x+xx,weights=ww));cy=float(np.average(y+yy,weights=ww))
            cov=np.cov(np.array([xx,yy]));eig=np.linalg.eigvalsh(cov)
            elong=float(np.sqrt((eig[1]+1)/(eig[0]+1)))
            bw=int(xx.max()-xx.min()+1);bh=int(yy.max()-yy.min()+1)
            extent=len(xx)/(bw*bh)
            if max(bw,bh)<35 or min(bw,bh)<16 or elong>5 or extent<.22:continue
            dist=float(bg['distance'][min(H-1,round(cy)),min(W-1,round(cx))])
            zone='plant' if dist<=50 and cy<FLOOR[group]-20 else 'bottom' if cy>=FLOOR[group]-80 else 'other'
            bbox=[int(x+xx.min()),int(y+yy.min()),bw,bh]
            detections.append(dict(x=round(cx,2),y=round(cy,2),bbox=bbox,area=int(len(xx)),peak=round(peak,4),elongation=round(elong,3),plant_distance=round(dist,2),
                                   zone=zone,side=('left' if cx<DIVIDER[group] else 'right') if zone=='plant' else '',
                                   extent=round(extent,3),suspect=bool(area>p['split_area'] or peak<.12 or dist>25 or min(bw,bh)<15 or y<5 or cy>FLOOR[group]-100)))
    detections.sort(key=lambda d:(d['x'],d['y']))
    return detections,dict(brightness=float(np.median(cv2.cvtColor(im[::8,::8],cv2.COLOR_BGR2GRAY))),
                          sharpness=float(cv2.Laplacian(cv2.cvtColor(im[::4,::4],cv2.COLOR_BGR2GRAY),cv2.CV_32F).var()))

def overlay(im,group,ds,scale=2048/W):
    canvas=cv2.resize(im,(round(W*scale),round(H*scale)))
    for j,d in enumerate(ds,1):
        x,y,w,h=[round(v*scale) for v in d['bbox']]
        color=(255,110,0) if d['side']=='left' else (0,100,255) if d['side']=='right' else (180,50,180)
        cv2.rectangle(canvas,(x-2,y-2),(x+w+2,y+h+2),color,1)
        cv2.putText(canvas,str(j),(x,y-3),cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
    cv2.line(canvas,(0,round(FLOOR[group]*scale)),(canvas.shape[1],round(FLOOR[group]*scale)),(210,90,210),1)
    return canvas

if __name__=='__main__':
    import argparse,time
    ap=argparse.ArgumentParser();ap.add_argument('--background',action='store_true');ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
    if args.background:
        for g in 'ABC':prepare_background(g);print('background',g,flush=True)
    if args.pilot:
        rows=[]
        for g in 'ABC':
            for j in [1,11,20]:
                sid=f'{g}_cal_{j:02d}';im,_=read(OUT/'cache/samples'/f'{sid}.JPG');start=time.monotonic();ds,q=detect(im,g)
                row=dict(sample_id=sid,group=g,left=sum(d['side']=='left' for d in ds),right=sum(d['side']=='right' for d in ds),other=sum(d['zone']!='plant' for d in ds),detections=ds,**q)
                rows.append(row);cv2.imwrite(str(OUT/'qa'/f'{sid}_prototype.jpg'),overlay(im,g,ds));print(sid,row['left'],row['right'],row['other'],'seconds',round(time.monotonic()-start,3),flush=True)
        (OUT/'data/pilot_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
