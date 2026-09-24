"""Locally learned insect-shape filter, with no downloaded weights."""
from pathlib import Path
import json,time,cv2,numpy as np
from sklearn.svm import LinearSVC
from joblib import dump,load
import vision as v
OUT=v.OUT
cv2.setNumThreads(1)

def hog():return cv2.HOGDescriptor((64,128),(16,16),(8,8),(8,8),9,1,-1,0,.2,True,8,False)

def patch(im,x,y,w=78,h=144):
    x0=max(0,int(x-w/2));y0=max(0,int(y-h/2));x1=min(im.shape[1],int(x+w/2));y1=min(im.shape[0],int(y+h/2))
    return cv2.resize(im[y0:y1,x0:x1],(64,128))

def train():
    samples=json.loads((OUT/'data/single_body_training_points.json').read_text())['samples']
    rng=np.random.default_rng(160916);hh=hog();xx=[];yy=[];start=time.monotonic()
    for sid,entry in samples.items():
        im,_=v.read(OUT/'cache/samples'/f'{sid}.JPG');im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        points=np.array(entry['points'],float)*v.W/entry['width']
        for x,y in points:
            for k in range(16):
                fac=rng.uniform(.85,1.15);p=patch(im,x+rng.uniform(-6,6),y+rng.uniform(-9,9),78*fac,144*fac)
                if k%2:p=cv2.flip(p,1)
                if k%4>=2:p=cv2.flip(p,0)
                xx.append(hh.compute(p).ravel());yy.append(1)
        # Negatives are selected outside all broad candidate regions, rather
        # than treating unlabelled insect-like regions as known background.
        original,_=v.read(OUT/'cache/samples'/f'{sid}.JPG');hp=v.contrast(original)
        proposals=(hp>.15).astype(np.uint8)
        proposals=cv2.dilate(proposals,np.ones((31,31),np.uint8))
        added=0
        for k in range(7000):
            x=int(rng.uniform(120,v.W-120));y=int(rng.uniform(100,v.FLOOR[sid[0]]-100))
            if np.any((np.abs(points[:,0]-x)<100)&(np.abs(points[:,1]-y)<160)):continue
            if proposals[y-25:y+25,x-15:x+15].mean()>.1:continue
            xx.append(hh.compute(patch(im,x,y)).ravel());yy.append(0);added+=1
            if added>=600:break
        print('training crops',sid,added,flush=True)
    # First complementation image is visually almost empty on the plants and
    # supplies pale-green stem/leaf negatives, excluding the two wall insects.
    im,_=v.read(OUT/'cache/samples/C_cal_01.JPG');im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    for k in range(1200):
        x=int(rng.uniform(400,3500));y=int(rng.uniform(100,2050))
        xx.append(hh.compute(patch(im,x,y)).ravel());yy.append(0)
    # Explicitly reviewed leaf tips, sheath fragments, and box textures.
    negative_file=OUT/'data/reviewed_negative_boxes.json'
    if negative_file.exists():
        for item in json.loads(negative_file.read_text()):
            im,_=v.read(OUT/'cache/samples'/f"{item['sample_id']}.JPG");im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            bx,by,bw,bh=item['bbox']
            for k in range(48):
                factor=rng.uniform(.9,1.1)
                a=patch(im,bx+bw/2+rng.uniform(-8,8),by+bh/2+rng.uniform(-10,10),bw*factor,bh*factor)
                if k%2:a=cv2.flip(a,1)
                xx.append(hh.compute(a).ravel());yy.append(0)
    x=np.asarray(xx,np.float32);y=np.array(yy)
    for round_index in range(3):
        clf=LinearSVC(C=.015,class_weight='balanced',max_iter=15000,dual='auto',random_state=160916).fit(x,y)
        hh.setSVMDetector(np.r_[clf.coef_.ravel(),clf.intercept_].astype(np.float32))
        hard=[]
        for sid,entry in {**samples,'C_cal_01':{'points':[],'width':2000}}.items():
            original,_=v.read(OUT/'cache/samples'/f'{sid}.JPG');gray=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
            points=np.asarray(entry['points'],float).reshape(-1,2)*v.W/entry['width']
            boxes,scores=hh.detectMultiScale(gray[:v.FLOOR[sid[0]]-100],hitThreshold=-.1,winStride=(8,8),scale=1.06,groupThreshold=0)
            hp=v.contrast(original)
            for i in np.argsort(np.asarray(scores).ravel())[::-1][:6000]:
                bx,by,bw,bh=map(int,boxes[i]);cx=bx+bw/2;cy=by+bh/2
                if np.any((np.abs(points[:,0]-cx)<110)&(np.abs(points[:,1]-cy)<160)):continue
                if sid=='C_cal_01':
                    if not (400<cx<3600 and cy<2050):continue
                elif np.quantile(hp[by:by+bh,bx:bx+bw],.98)>.12:continue
                hard.append(hh.compute(cv2.resize(gray[by:by+bh,bx:bx+bw],(64,128))).ravel())
        if hard:
            x=np.concatenate([x,np.asarray(hard,np.float32)]);y=np.r_[y,np.zeros(len(hard),int)]
        print('hard negative round',round_index+1,len(hard),flush=True)
    clf=LinearSVC(C=.015,class_weight='balanced',max_iter=15000,dual='auto',random_state=160916).fit(x,y)
    np.savez_compressed(OUT/'cache/hog_training.npz',x=x,y=y)
    coef=np.r_[clf.coef_.ravel(),clf.intercept_].astype(np.float32)
    np.save(OUT/'cache/hog_weights.npy',coef);dump(clf,OUT/'cache/hog_model.joblib')
    print('trained',len(y),'positive',sum(y),'training_accuracy',clf.score(x,y),'seconds',time.monotonic()-start,flush=True)

def detect(im,group,hit=0):
    hh=hog();hh.setSVMDetector(np.load(OUT/'cache/hog_weights.npy'))
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    boxes,scores=hh.detectMultiScale(gray[:v.FLOOR[group]+20],hitThreshold=hit,winStride=(8,8),padding=(0,0),scale=1.06,groupThreshold=0,useMeanshiftGrouping=False)
    ds=[];bg=v.background(group)
    for i in np.argsort(np.array(scores).ravel())[::-1]:
        x,y,w,h=map(int,boxes[i]);cx=x+w/2;cy=y+h/2
        if any(((cx-d['x'])/45)**2+((cy-d['y'])/65)**2<1 for d in ds):continue
        dist=float(bg['distance'][min(v.H-1,round(cy)),min(v.W-1,round(cx))])
        zone='plant' if dist<=55 and cy<v.FLOOR[group]-20 else 'bottom' if cy>=v.FLOOR[group]-80 else 'other'
        ds.append(dict(x=cx,y=cy,bbox=[x,y,w,h],score=float(scores[i]),side=('left' if cx<v.DIVIDER[group] else 'right') if zone=='plant' else '',zone=zone,plant_distance=dist,suspect=float(scores[i])<.5))
    ds.sort(key=lambda d:(d['x'],d['y']))
    return ds

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--train',action='store_true');args=p.parse_args()
    if args.train:train()
    rows=[]
    for g in 'ABC':
        for j in [1,11,20]:
            sid=f'{g}_cal_{j:02d}';im,_=v.read(OUT/'cache/samples'/f'{sid}.JPG');start=time.monotonic();ds=detect(im,g,.35)
            rows.append(dict(sample_id=sid,detections=ds));cv2.imwrite(str(OUT/'qa'/f'{sid}_hog.jpg'),v.overlay(im,g,ds))
            print(sid,sum(d['side']=='left' for d in ds),sum(d['side']=='right' for d in ds),sum(not d['side'] for d in ds),'seconds',round(time.monotonic()-start,2),flush=True)
    (OUT/'data/hog_pilot.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
