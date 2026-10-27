"""Run an edited Python script on CUDA; generate its @fast kernels locally.

The GPU worker receives source and kernels. Model credentials stay on the controller.
"""
from __future__ import annotations
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time


def run(script, out, remote, remote_dir, max_usd=1.0, minutes=30, model='muse-spark-1.3-contributor'):
    script=Path(script).resolve();out=Path(out).resolve();prefix=shlex.split(remote)
    if not script.is_file():raise ValueError('Provide a Python script')
    if not 0<max_usd<=25 or not 0<minutes<=240:raise ValueError('Invalid budget')
    out.mkdir(parents=True,exist_ok=False)
    source=script.read_bytes();(out/'script.py').write_bytes(source)
    deadline=time.monotonic()+minutes*60
    def left():
        remaining=int(deadline-time.monotonic())
        if remaining<1:raise TimeoutError('Remote run exceeded its time budget')
        return remaining
    def command(cmd,input=None):
        wrapped=f'timeout --kill-after=10s {left()}s bash -c {shlex.quote(cmd)}'
        return subprocess.run(prefix+[wrapped],input=input,capture_output=True,check=True,timeout=left()+15)
    def upload(path,content):
        command(f'mkdir -p {shlex.quote(str(Path(path).parent))} && cat > {shlex.quote(path+".tmp")} && mv {shlex.quote(path+".tmp")} {shlex.quote(path)}',content)
    def download_files(path,names):
        code='import pathlib,json,base64; p=pathlib.Path('+repr(path)+'); print(json.dumps({n:base64.b64encode((p/n).read_bytes()).decode() for n in '+repr(list(names))+' if (p/n).is_file()}))'
        return {n:base64.b64decode(value) for n,value in json.loads(command('python -c '+shlex.quote(code)).stdout).items()}
    def upload_files(path,files):
        code='import pathlib,json,base64,sys; p=pathlib.Path('+repr(path)+'); p.mkdir(parents=True,exist_ok=True)\nfor n,v in json.load(sys.stdin).items():\n t=p/(n+".tmp"); t.write_bytes(base64.b64decode(v)); t.replace(p/n)'
        command('python -c '+shlex.quote(code),json.dumps({n:base64.b64encode(v).decode() for n,v in files.items()}).encode())
    key=hashlib.sha256((shlex.join(prefix)+' '+remote_dir).encode()).hexdigest()[:20]
    lock=open(Path(tempfile.gettempdir())/('recurrence-worker-'+key+'.lock'),'a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another experiment is using this GPU worker') from None
    run_id=out.name+'_'+hashlib.sha256(str(out).encode()).hexdigest()[:8]
    worker=f'{remote_dir}/muse_runs/{run_id}';cache=f'{remote_dir}/.muse_kernels/remote'
    config={'script':script.name,'script_sha256':hashlib.sha256(source).hexdigest(),'model':model,'model_budget_usd':max_usd,'wall_minutes':minutes}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    ledger=out/'budget.json';ledger.write_text(json.dumps({'usd':0.0,'reservations':{}}))
    env={**os.environ,'MUSE_EXPERIMENT_LEDGER':str(ledger),'MUSE_EXPERIMENT_BUDGET_USD':str(max_usd),
         'MUSE_KERNEL_REMOTE':remote,'MUSE_KERNEL_REMOTE_DIR':remote_dir}
    proc=None
    try:
        upload(worker+'/script.py',source)
        launch=shlex.join(['env','PYTHONPATH='+remote_dir,'MUSE_KERNEL_EXTERNAL_BUILDER=1','MUSE_KERNEL_CACHE='+cache,
                          'MUSE_KERNEL_MODEL='+model,'MUSE_KERNEL_MAX_USD='+str(max_usd),'python','-u',worker+'/script.py'])
        wrapped=f'echo $$ > {shlex.quote(worker+"/process.pid")}; exec timeout --kill-after=10s {left()}s bash -c '+shlex.quote(f'cd {shlex.quote(remote_dir)} && {launch}')
        with open(out/'training.log','w') as log:
            proc=subprocess.Popen(prefix+[wrapped],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            handled=set()
            while proc.poll() is None:
                query='import json,pathlib; p=pathlib.Path('+repr(cache)+'); print(json.dumps([str(x.parent) for x in p.glob("*/status.json") if json.loads(x.read_text()).get("state")=="queued"]))'
                queued=json.loads(command('python -c '+shlex.quote(query)).stdout)
                requested=set()
                for line in (out/'training.log').read_text().splitlines():
                    if line.startswith('[muse_kernel] queued '):requested.add(line.split('queued ',1)[1])
                    try:
                        event=json.loads(line)
                        if event.get('phase')=='captured':requested.add(event.get('cache'))
                    except ValueError:pass
                for path in queued:
                    if path not in requested or path in handled:continue
                    handled.add(path);job=out/Path(path).name;job.mkdir()
                    for name,content in download_files(path,('job.json','reference.py','spec.json','extra_specs.json','status.json')).items():
                        (job/name).write_bytes(content)
                    settings=json.loads((job/'job.json').read_text())
                    settings.update(total_budget=True,opt_rounds=0,bwd_opt_rounds=0)
                    (job/'job.json').write_text(json.dumps(settings,indent=2))
                    upload(path+'/status.json',json.dumps({'state':'building'}).encode())
                    print('Building '+settings['name']+' · '+settings['hash'],flush=True)
                    with open(job/'controller.log','w') as build_log:
                        child=subprocess.Popen([sys.executable,'-m','muse_kernel.build',str(job)],
                            env={**env,'MUSE_KERNEL_REMOTE_TIMEOUT':str(left())},stdout=build_log,stderr=subprocess.STDOUT,start_new_session=True)
                        try:child.wait(timeout=left())
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=10);raise
                    state=json.loads((job/'status.json').read_text())
                    if state['state']=='ready' and not state.get('holdout_passed'):
                        # Recheck the selected bytes with seeds never shown to the writer.
                        from kernel_contract.holdout import holdout_cases
                        from kernel_contract.evidence import coverage,sha256
                        protocol=holdout_cases(None);protocol['candidate_sha256']=sha256(job/'kernel.py')
                        (job/'holdout_protocol.json').write_text(json.dumps(protocol,indent=2))
                        remote_job=f'{remote_dir}/muse_jobs/{settings["hash"]}'
                        upload(remote_job+'/selected.py',(job/'kernel.py').read_bytes())
                        upload(remote_job+'/holdout.json',json.dumps(protocol).encode())
                        check=command(f'cd {shlex.quote(remote_dir)} && '+shlex.join(['python','-m','muse_kernel.remote_eval',remote_job,remote_job+'/selected.py','--grads','--holdout',remote_job+'/holdout.json']))
                        result=json.loads(next(l[7:] for l in check.stdout.decode().splitlines() if l.startswith('RESULT ')))
                        result['coverage']=coverage(result['results']);(job/'holdout_results.json').write_text(json.dumps(result,indent=2))
                        if not result['coverage']['all_cases_passed']:
                            state.update(state='failed',reason='Fresh holdout failed; kernel withheld')
                            (job/'status.json').write_text(json.dumps(state,indent=2))
                        else:
                            upload_files(path,{name:(job/name).read_bytes() for name in ('kernel.py','checks.json','holdout_protocol.json','holdout_results.json')})
                    if state['state']=='ready' and state.get('holdout_passed'):
                        upload_files(path,{name:(job/name).read_bytes() for name in ('kernel.py','checks.json','holdout_protocol.json','holdout_results.json')})
                    upload(path+'/status.json',json.dumps(state).encode())
                    print(settings['name']+': '+state['state'],flush=True)
                time.sleep(min(2,left()))
        # Retain the contract and receipts even when a matching committed cache was reused.
        for line in (out/'training.log').read_text().splitlines():
            try:record=json.loads(line)
            except ValueError:continue
            if record.get('phase')!='captured':continue
            path=record.get('cache','')
            if not path.startswith(cache+'/') or '/' in path[len(cache)+1:]:continue
            job=out/Path(path).name;job.mkdir(exist_ok=True)
            names=('job.json','reference.py','spec.json','extra_specs.json','status.json','kernel.py','checks.json','holdout_protocol.json','holdout_results.json')
            for name,content in download_files(path,names).items():
                if not (job/name).exists():(job/name).write_bytes(content)
        if proc.returncode:raise RuntimeError('Remote script failed; inspect '+str(out/'training.log'))
        print('Finished. '+str(out/'training.log'),flush=True)
    finally:
        if proc and proc.poll() is None:
            cleanup='import os,pathlib,signal; p=pathlib.Path('+repr(worker+'/process.pid')+'); pid=int(p.read_text()) if p.exists() else 0; os.kill(pid,signal.SIGTERM) if pid>1 else None'
            subprocess.run(prefix+['python -c '+shlex.quote(cleanup)],capture_output=True,timeout=15)
            os.killpg(proc.pid,signal.SIGTERM)
            proc.wait(timeout=15)
        lock.close()
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('script');ap.add_argument('--out',required=True);ap.add_argument('--remote',required=True)
    ap.add_argument('--remote-dir',required=True);ap.add_argument('--max-usd',type=float,default=1)
    ap.add_argument('--minutes',type=int,default=30);ap.add_argument('--model',default='muse-spark-1.3-contributor')
    a=ap.parse_args();run(a.script,a.out,a.remote,a.remote_dir,a.max_usd,a.minutes,a.model)

if __name__=='__main__':main()
