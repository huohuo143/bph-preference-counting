"""Local-only original-image and count review. Never writes to source images."""
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
from datetime import datetime
import csv,json,sqlite3,threading

OUT=Path(__file__).resolve().parents[1]
LOCK=threading.Lock()

def load_rows():
    with (OUT/'data/逐图计数与质量.csv').open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

class Handler(BaseHTTPRequestHandler):
    def send_bytes(self,data,mime,status=200):
        self.send_response(status);self.send_header('Content-Type',mime)
        self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
    def send_json(self,data,status=200):self.send_bytes(json.dumps(data,ensure_ascii=False).encode(),'application/json; charset=utf-8',status)
    def do_GET(self):
        p=urlparse(self.path);q=parse_qs(p.query)
        if p.path=='/':return self.send_bytes((OUT/'review/照片复核.html').read_bytes(),'text/html; charset=utf-8')
        if p.path=='/api/frame':
            g=q.get('group',['A'])[0];group=[r for r in self.server.rows if r['group']==g]
            try:i=max(0,min(len(group)-1,int(q.get('index',['0'])[0])))
            except ValueError:return self.send_json({'error':'照片序号无效'},400)
            if not group:return self.send_json({'error':'没有该组'},400)
            row=group[i]
            with sqlite3.connect(f'file:{OUT}/data/自动检测索引.sqlite?mode=ro',uri=True) as db:
                raw=db.execute('SELECT payload FROM frames WHERE frame_id=?',(row['frame_id'],)).fetchone()
            payload=json.loads(raw[0]) if raw else {}
            return self.send_json({'row':row,'index':i,'total':len(group),'detections':payload.get('detections',[]),'review':self.server.reviews.get(row['frame_id'])})
        if p.path=='/image':
            fid=q.get('frame_id',[''])[0];row=self.server.byid.get(fid)
            if not row:return self.send_json({'error':'照片不存在'},404)
            try:return self.send_bytes(Path(row['source_path']).read_bytes(),'image/jpeg')
            except OSError:return self.send_json({'error':'原图不可读，请检查原始数据卷是否挂载'},404)
        return self.send_json({'error':'未找到'},404)
    def do_POST(self):
        # A cross-origin web page must not be able to append a false review.
        origin=self.headers.get('Origin')
        if origin and origin not in [f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}']:
            return self.send_json({'error':'仅允许本机复核页面保存'},403)
        if self.path!='/api/review':return self.send_json({'error':'未找到'},404)
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<1000000:raise ValueError('请求大小无效')
            d=json.loads(self.rfile.read(n));fid=d['frame_id']
            if fid not in self.server.byid:raise ValueError('未知照片')
            if not d.get('reviewer','').strip():raise ValueError('请填写复核者')
            if d.get('confirmed') is not True:raise ValueError('请先核对整张照片并确认')
            boxes=d.get('boxes')
            if not isinstance(boxes,list) or len(boxes)>1000:raise ValueError('标注无效')
            for b in boxes:
                if b.get('kind') not in ['left','right','other','suspect']:raise ValueError('类别无效')
                x,y,w,h=map(float,b['bbox'])
                if not(0<=x<4416 and 0<=y<2488 and 0<w<=4416-x and 0<h<=2488-y):raise ValueError('坐标超出原图')
            d.update(saved_at=datetime.now().astimezone().isoformat(),source_path=self.server.byid[fid]['source_path'],
                source_sha256=self.server.byid[fid]['source_sha256'],review_source='local_user_interface',
                left_count=sum(b['kind']=='left' for b in boxes),right_count=sum(b['kind']=='right' for b in boxes),
                other_count=sum(b['kind']=='other' for b in boxes),suspect_count=sum(b['kind']=='suspect' for b in boxes))
            dest=OUT/'review/人工复核追加记录.jsonl'
            with LOCK,dest.open('a') as f:
                f.write(json.dumps(d,ensure_ascii=False)+'\n');self.server.reviews[fid]=d
            return self.send_json({'saved':True,'file':str(dest),'message':'复核记录已追加保存，原图和当前统计未改写。'})
        except (ValueError,KeyError,TypeError) as exc:return self.send_json({'error':str(exc)},400)

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8768);p.add_argument('--open-browser',action='store_true');a=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    server.rows=load_rows();server.byid={r['frame_id']:r for r in server.rows};server.reviews={}
    path=OUT/'review/人工复核追加记录.jsonl'
    if path.exists():
        for line in path.open():
            r=json.loads(line);server.reviews[r['frame_id']]=r
    print(f'本机照片复核：http://127.0.0.1:{a.port}；Ctrl+C结束。',flush=True)
    if a.open_browser:
        import webbrowser
        threading.Timer(.5,lambda:webbrowser.open(f'http://127.0.0.1:{a.port}')).start()
    server.serve_forever()
if __name__=='__main__':main()
