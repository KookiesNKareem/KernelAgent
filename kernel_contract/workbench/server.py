"""Local researcher workbench. One active experiment; no remotely supplied Python or shell."""
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from .experiment import validate
from .variants import DEFAULT_VARIANTS, variant, templates

PAGE=Path(__file__).with_name('index.html')
ROOT=Path(__file__).resolve().parents[2]


def json_lines(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text().splitlines():
        try:result.append(json.loads(line))
        except ValueError:pass  # a writer may be appending the final line
    return result


def recorded():
    from .report import training_summary, paired_summary
    from kernel_contract.evidence import coverage
    path=ROOT/'runs_public/triadic/wonder/events.jsonl'
    if not path.exists():return {'mode':'recorded','unavailable':True}
    events=json_lines(path)
    done=next(e for e in reversed(events) if e['event']=='done')
    selected=next(e for e in events if e['event']=='eval' and e['path']==done['backward'])
    elapsed=events[-1]['t']-events[0]['t']
    ref=training_summary(ROOT/'runs_public/train_triadic/fig_ckpt.jsonl')
    kernel=training_summary(ROOT/'runs_public/train_triadic/fig_kernel.jsonl')
    return {'mode':'recorded','label':'Recorded Triadic experiment · October 27, 2026 · L4',
            'reference_source': (ROOT/'kernel_contract/layers/triadic_gdn.py').read_text(), 'events':events,'summary':{'results':[{'name':'Triadic Gated DeltaNet','state':'complete',
            'generation_s':elapsed,'model_usd':done['run_usd'],'coverage':coverage(selected['results']),
            'fwd_bwd_ms':done['fwd_bwd_ms'],'pairs':[{'seed':0,'reference':ref,'kernel':kernel,
                **paired_summary(ref,kernel,elapsed)}]}], 'budget':{'usd':done['run_usd'],'reserved_usd':0},
            'claim':'Historical single-seed run; one validation batch. Fresh workbench runs use multiple batches.'}}


def inspect_run(path):
    events=json_lines(path/'events.jsonl')
    builds=[]
    for p in sorted(path.glob('variant_*/run*/events.jsonl')):
        for e in json_lines(p):
            builds.append({**e,'variant':p.parent.parent.name})
    summary=json.loads((path/'summary.json').read_text()) if (path/'summary.json').exists() else None
    config=json.loads((path/'config.json').read_text())
    contracts=[json.loads(p.read_text()) if p.exists() else None for p in (path/f'variant_{i+1}/spec.json' for i in range(len(config['variants'])))]
    return {'mode':'live','id':path.name,'config':config,'contracts':contracts,'events':events,'build_events':builds,'summary':summary,
            'bundle_ready':(path/'reproducible-bundle.zip').exists()}


def make_handler(root, remote, remote_dir, token, offline=False):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    state={'process':None,'run':None};lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def respond(self,value,status=200,kind='application/json'):
            body=value if isinstance(value,bytes) else json.dumps(value).encode()
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers();self.wfile.write(body)
        def trusted(self):
            host=self.headers.get('Host','')
            return host in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
        def do_GET(self):
            if not self.trusted():return self.respond({'error':'Local access only'},403)
            path=urlparse(self.path).path
            if path=='/':return self.respond(PAGE.read_text().replace('__TOKEN__',token).encode(),kind='text/html; charset=utf-8')
            if path=='/api/config':return self.respond({'variants':DEFAULT_VARIANTS,'templates':templates(),'gpu_configured':bool(remote_dir) and not offline,'active':state['run'] if state['process'] and state['process'].poll() is None else None})
            if path=='/api/recorded':return self.respond(recorded())
            if path=='/api/runs':return self.respond({'runs':[p.name for p in sorted(root.iterdir(),reverse=True) if p.is_dir() and (p/'config.json').exists()]})
            m=re.fullmatch(r'/api/runs/([a-zA-Z0-9_-]+)(/bundle)?',path)
            if m:
                p=root/m[1]
                if not p.is_dir():return self.respond({'error':'Unknown experiment'},404)
                if m[2]:
                    z=p/'reproducible-bundle.zip'
                    if not z.exists():return self.respond({'error':'Bundle not ready'},409)
                    return self.respond(z.read_bytes(),kind='application/zip')
                return self.respond(inspect_run(p))
            return self.respond({'error':'Not found'},404)
        def do_POST(self):
            if not self.trusted() or self.headers.get('X-Workbench-Token')!=token:return self.respond({'error':'Local session token required'},403)
            origin=self.headers.get('Origin')
            if origin and urlparse(origin).netloc!=self.headers.get('Host'):return self.respond({'error':'Origin mismatch'},403)
            try:
                size=int(self.headers.get('Content-Length',0))
                if not 0<size<=100000:raise ValueError('Invalid request size')
                data=json.loads(self.rfile.read(size))
                if self.path=='/api/preview':
                    import difflib
                    vs=[variant(v) for v in data['variants']]
                    for v in vs:
                        v['diff']=''.join(difflib.unified_diff(vs[0]['source'].splitlines(True),v['source'].splitlines(True),fromfile=vs[0]['name'],tofile=v['name']))
                    return self.respond({'variants':vs})
                if self.path!='/api/run':return self.respond({'error':'Not found'},404)
                if offline:return self.respond({'error':'GPU worker is stopped. Restart the workbench with a configured worker to run an experiment.'},409)
                c=validate(data)
                with lock:
                    if state['process'] and state['process'].poll() is None:return self.respond({'error':'An experiment is already running'},409)
                    import datetime
                    name=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(3)
                    config=root/(name+'.json');config.write_text(json.dumps(c,indent=2))
                    out=root/name
                    cmd=[sys.executable,'-m','kernel_contract.workbench.experiment','--config',str(config),'--out',str(out),
                         '--remote',remote,'--remote-dir',remote_dir]
                    with open(root/(name+'.log'),'w') as f:
                        state['process']=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    state['run']=name
                return self.respond({'id':name},202)
            except (ValueError,KeyError,TypeError,SyntaxError) as e:return self.respond({'error':str(e)},400)
    return Handler


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8765)
    ap.add_argument('--runs',default='runs/workbench');ap.add_argument('--remote',default=os.environ.get('MUSE_KERNEL_REMOTE','bash -c'))
    ap.add_argument('--remote-dir',default=os.environ.get('MUSE_KERNEL_REMOTE_DIR',str(ROOT)))
    ap.add_argument('--offline',action='store_true',help='Inspect and edit without starting GPU experiments')
    a=ap.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),make_handler(a.runs,a.remote,a.remote_dir,secrets.token_urlsafe(32),offline=a.offline))
    print(f'http://127.0.0.1:{a.port}',flush=True)
    server.serve_forever()

if __name__=='__main__':main()
