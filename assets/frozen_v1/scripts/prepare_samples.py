from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,hashlib,time
import cv2,numpy as np
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parents[1]
cv2.setNumThreads(1)

def prepare(r):
    folder=OUT/'cache/samples';folder.mkdir(exist_ok=True)
    p=folder/(r['sample_id']+'.JPG')
    if not p.exists():p.write_bytes(Path(r['source_path']).read_bytes())
    raw=p.read_bytes(); im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    small=cv2.resize(im,(2000,round(im.shape[0]*2000/im.shape[1])),interpolation=cv2.INTER_AREA)
    preview=OUT/'qa'/f'{r["sample_id"]}_preview.jpg'
    cv2.imwrite(str(preview),small,[cv2.IMWRITE_JPEG_QUALITY,95])
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    r=dict(r);r.update(sha256=hashlib.sha256(raw).hexdigest(),brightness=float(np.median(gray)),sharpness=float(cv2.Laplacian(gray[::4,::4],cv2.CV_32F).var()))
    return r

def main():
    rows=json.loads((OUT/'data/sample_selection.json').read_text());start=time.monotonic()
    with ThreadPoolExecutor(max_workers=6) as pool:rs=list(pool.map(prepare,rows))
    (OUT/'data/sample_manifest.json').write_text(json.dumps(rs,ensure_ascii=False,indent=2))
    for group in 'ABC':
        a=[r for r in rs if r['group']==group and r['split']=='calibration']
        canvas=Image.new('RGB',(2400,1840),'white');d=ImageDraw.Draw(canvas)
        for j,r in enumerate(a):
            with Image.open(OUT/'qa'/f'{r["sample_id"]}_preview.jpg') as im:
                im=im.resize((600,338));x=j%4*600;y=j//4*368;canvas.paste(im,(x,y+28))
                d.text((x+8,y+5),f'{r["sample_id"]} | frame {r["sequence_index"]} | brightness {r["brightness"]:.0f}',fill='black')
        canvas.save(OUT/'qa'/f'{group}_calibration_overview.jpg',quality=92)
    print('prepared',len(rs),'seconds',round(time.monotonic()-start,1),flush=True)
if __name__=='__main__':main()
