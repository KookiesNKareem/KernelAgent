"""Run 1–3 researcher-defined variants under one model budget and wall-clock limit."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

from .variants import variant, DEFAULT_VARIANTS
from .report import training_summary, paired_summary, aggregate_pairs
from kernel_contract.evidence import sha256, coverage

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def validate(config):
    variants = config.get('variants', DEFAULT_VARIANTS)
    if not 1 <= len(variants) <= 3:
        raise ValueError('Choose one to three variants')
    vs = [variant(v) for v in variants]
    if len({v['hash'] for v in vs}) != len(vs):
        raise ValueError('Each variant must have a distinct equation')
    settings = {'model_budget_usd': 3.0, 'wall_minutes': 60, 'models': ['muse-spark-1.3-contributor'],
                'steps': 100, 'seeds': [0, 1, 2], 'val_batches': 8, 'batch': 2, 'seq': 128,
                'heads': 2, 'head_dim': 32, 'width': 128, 'layers': 2, 'target_loss': 2.5,
                'gpu_hourly_usd': None, 'direct_control': False}
    settings.update({k:config[k] for k in settings if k in config})
    import math
    for key in ('model_budget_usd', 'wall_minutes'):
        if not isinstance(settings[key], (float,int)) or not math.isfinite(settings[key]) or not 0 < settings[key] <= (25 if key == 'model_budget_usd' else 240):
            raise ValueError(f'{key} is outside the supported budget range')
    for key in ('steps','val_batches','batch','seq','heads','head_dim','width','layers'):
        if type(settings[key]) != int or not 1 <= settings[key] <= {'steps':2000,'val_batches':32,'batch':16,'seq':2048,'heads':16,'head_dim':256,'width':1024,'layers':8}[key]:
            raise ValueError(f'Invalid {key}')
    if not isinstance(settings['seeds'], list) or not 1 <= len(settings['seeds']) <= 5 or any(type(s)!=int or not 0<=s<2**31 for s in settings['seeds']):
        raise ValueError('Provide 1–5 integer seeds')
    if len(set(settings['seeds'])) != len(settings['seeds']):
        raise ValueError('Seeds must be distinct')
    allowed = {'muse-spark-1.3-contributor','muse-wonder-1.0','meta-models/muse-spark-open-1026'}
    if not settings['models'] or any(m not in allowed for m in settings['models']):
        raise ValueError('Choose supported Meta models')
    rate=settings['gpu_hourly_usd']
    if rate is not None and (not isinstance(rate,(int,float)) or not math.isfinite(rate) or rate<0):
        raise ValueError('GPU hourly rate must be nonnegative')
    return {**settings, 'variants':vs, 'schema_version':1}


class Experiment:
    def __init__(self, config, out, remote='bash -c', remote_dir=None):
        self.config = validate(config)
        self.out = Path(out).resolve()
        self.remote = shlex.split(remote)
        self.remote_dir = remote_dir or str(ROOT)
        self.started = time.time()
        self.deadline = time.monotonic() + self.config['wall_minutes']*60
        self.results = []
        self.out.mkdir(parents=True, exist_ok=False)
        write_json(self.out/'config.json', self.config)
        from .bundle import snapshot_source
        snapshot_source(self.out/'.source_snapshot')
        self.ledger = self.out/'budget.json'
        write_json(self.ledger, {'usd':0.0,'reservations':{}})
        self.env = {**os.environ, 'MUSE_EXPERIMENT_LEDGER':str(self.ledger),
                    'MUSE_EXPERIMENT_BUDGET_USD':str(self.config['model_budget_usd']),
                    'MUSE_KERNEL_REMOTE':shlex.join(self.remote), 'MUSE_KERNEL_REMOTE_DIR':self.remote_dir}
        self.emit('experiment_start', mode='live', config=self.config)

    def emit(self, event, **values):
        e = {'event':event, 't':time.time(), **values}
        with open(self.out/'events.jsonl','a') as f:
            f.write(json.dumps(e)+'\n')
        print(json.dumps(e),flush=True)

    def remaining(self):
        left = int(self.deadline-time.monotonic())
        if left < 1:
            raise TimeoutError('Experiment wall-clock budget exhausted')
        return left

    def remote_run(self, command, input=None):
        # timeout is executed on the GPU host as well: an interrupted SSH connection must not leave work running.
        limit=self.remaining()
        wrapped=f'timeout --kill-after=10s {limit}s bash -c {shlex.quote(command)}'
        p=subprocess.run(self.remote+[wrapped], input=input, capture_output=True, timeout=limit+15)
        if p.returncode:
            detail=p.stderr.decode(errors='replace')[-1500:]
            raise subprocess.CalledProcessError(p.returncode,command,output=p.stdout,stderr=detail)
        return p

    def upload(self, dest, content):
        self.remote_run(f'mkdir -p {shlex.quote(str(Path(dest).parent))} && cat > {shlex.quote(dest)}', content)

    def prepare(self, v, job):
        import torch
        from kernel_contract.layers._common import make_inputs
        from kernel_contract.spec import capture, with_domains
        from kernel_contract.agent import describe_call
        c=self.config
        x=make_inputs(True,B=c['batch'],H=c['heads'],T=c['seq'],D=c['head_dim'],device='cpu')
        h=v['hash']; op=f'muse_jobs.j_{h}.reference'
        spec=with_domains(capture(op, (*x,c['head_dim']**-0.5)),{1:'l2row',3:'l2row',4:'logdecay',5:'unit',6:'unit'})
        spec.flags.update(any_float_output=True,normwise_rtol=2e-2)
        job.mkdir()
        (job/'reference.py').write_text(v['source'])
        write_json(job/'spec.json',spec.to_json());write_json(job/'extra_specs.json',[])
        write_json(job/'job.json',{'hash':h,'name':'reference','source':v['source'],
            'signature':'q, k, v, e, g, beta, gamma, scale', 'call':describe_call(spec),'sizes':spec.sizes,
            'grad_names':'q, k, v, e, g, beta, gamma', 'domains':['k and e have unit-norm rows','g <= 0','beta and gamma lie in [0,1]'],
            'models':c['models'],'max_usd':c['model_budget_usd']/len(c['variants'])/(2 if c['direct_control'] else 1),'total_budget':True,'opt_rounds':0,'bwd_opt_rounds':0,'repairs':3})
        write_json(job/'status.json',{'state':'queued'})
        return spec

    def build(self, job):
        env={**self.env,'MUSE_KERNEL_REMOTE_TIMEOUT':str(self.remaining())}
        with open(job/'driver.log','w') as log:
            p=subprocess.Popen([sys.executable,'-m','muse_kernel.build',str(job)],env=env,stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True)
            try:
                rc=p.wait(timeout=self.remaining())
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=15)
                raise TimeoutError('Generation exceeded experiment wall-clock budget')
        state=json.loads((job/'status.json').read_text())
        if rc or state['state']!='ready':
            raise RuntimeError(state.get('reason','generation did not produce an accepted kernel'))
        return state

    def check_holdout(self, job, spec, v):
        from kernel_contract.holdout import holdout_cases
        protocol=holdout_cases(spec)
        protocol['candidate_sha256']=sha256(job/'kernel.py')
        write_json(job/'holdout_protocol.json',protocol)
        rjob=f'{self.remote_dir}/muse_jobs/{v["hash"]}'
        self.upload(rjob+'/selected.py',(job/'kernel.py').read_bytes())
        self.upload(rjob+'/holdout.json',json.dumps(protocol).encode())
        cmd=shlex.join(['python','-m','muse_kernel.remote_eval',rjob,rjob+'/selected.py','--grads','--holdout',rjob+'/holdout.json'])
        p=self.remote_run(f'cd {shlex.quote(self.remote_dir)} && {cmd}')
        result=json.loads(next(x[7:] for x in p.stdout.decode().splitlines() if x.startswith('RESULT ')))
        result['coverage']=coverage(result['results']);result['candidate_sha256']=protocol['candidate_sha256']
        write_json(job/'holdout_results.json',result)
        self.emit('holdout_done',variant=v['name'],coverage=result['coverage'],sha256=protocol['candidate_sha256'])
        if not result['coverage']['all_cases_passed']:
            raise RuntimeError('Selected candidate failed fresh holdout; withheld from training, no repair feedback')

    def train(self, job, v, mixer, seed):
        c=self.config; rjob=f'{self.remote_dir}/muse_jobs/{v["hash"]}'
        out=f'{rjob}/{mixer}_{seed}.jsonl'
        args=['python','-m','kernel_contract.train_demo','--layer',f'muse_jobs.j_{v["hash"]}',
              '--mixer',rjob+'/selected.py' if mixer=='kernel' else 'checkpointed','--out',out,'--seed',str(seed)]
        for name in ('steps','val_batches','batch','seq','heads','head_dim','width','layers'):
            args += ['--'+name.replace('_','-'),str(c[name])]
        self.emit('training_start',variant=v['name'],mixer=mixer,seed=seed)
        failed=None
        try:
            self.remote_run(f'cd {shlex.quote(self.remote_dir)} && {shlex.join(args)} > {shlex.quote(out+".log")} 2>&1')
        except subprocess.CalledProcessError as e:
            failed=e
        for suffix in ('','.meta.json'):
            p=self.remote_run(f'cat {shlex.quote(out+suffix)}')
            (job/f'{mixer}_{seed}.jsonl{suffix}').write_bytes(p.stdout)
        dataset=self.out/'dataset.txt'
        if not dataset.exists():dataset.write_bytes(self.remote_run('cat /tmp/tinyshakespeare.txt').stdout)
        receipt=json.loads((job/f'{mixer}_{seed}.jsonl.meta.json').read_text())
        if sha256(dataset)!=receipt['dataset_sha256']:
            raise RuntimeError('Dataset bytes changed during the experiment')
        summary=training_summary(job/f'{mixer}_{seed}.jsonl',c['target_loss'])
        if failed and not summary['out_of_memory']:
            raise RuntimeError(f'{mixer} training failed (see GPU log {out}.log)')
        self.emit('training_done',variant=v['name'],mixer=mixer,seed=seed,summary=summary)
        return summary

    def run(self):
        # Serialize all CLI/UI work for this configured worker to avoid overlapping benchmarks.
        import fcntl,tempfile
        key=hashlib.sha256((shlex.join(self.remote)+' '+self.remote_dir).encode()).hexdigest()[:20]
        lock=open(Path(tempfile.gettempdir())/('recurrence-worker-'+key+'.lock'),'a')
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            message='Another experiment is using this worker'
            self.emit('experiment_failed',error=message)
            self.save(error=message)
            self.emit('experiment_done',archive='reproducible-bundle.zip',results=[])
            from .bundle import export_bundle
            export_bundle(self.out)
            return []
        try:
            p=self.remote_run(f'cd {shlex.quote(self.remote_dir)} && python -c '+shlex.quote(
                'import json,torch,importlib.metadata; from muse_kernel.identity import environment; assert torch.cuda.is_available(), "CUDA is required"; e=environment(); e["packages"]={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}; print(json.dumps(e))'))
            write_json(self.out/'environment.json',json.loads(p.stdout))
            for i,v in enumerate(self.config['variants']):
                job=self.out/f'variant_{i+1}'
                row={'name':v['name'],'hash':v['hash'],'state':'building','pairs':[]}
                self.results.append(row)
                self.emit('variant_start',variant=v['name'],index=i+1)
                start=time.monotonic()
                try:
                    spec=self.prepare(v,job)
                    state=self.build(job)
                    row.update(generation_s=time.monotonic()-start, model_usd=state['usd'], coverage=state['coverage'],
                               kernel_sha256=state['kernel_sha256'])
                    self.emit('candidate_selected',variant=v['name'],**row)
                    self.check_holdout(job,spec,v)
                    row['build_and_check_s']=time.monotonic()-start
                    row['state']='training'
                    for seed in self.config['seeds']:
                        # Alternate order across seeds to reduce consistent thermal/order bias.
                        summaries={}
                        for mixer in (('reference','kernel') if seed%2==0 else ('kernel','reference')):
                            summaries[mixer]=self.train(job,v,mixer,seed)
                        row['pairs'].append({'seed':seed,**summaries,**paired_summary(summaries['reference'],summaries['kernel'],row['build_and_check_s'])})
                    row['state']='complete'
                    if self.config['direct_control']:
                        control=job/'direct_control'
                        self.prepare(v,control)
                        j=json.loads((control/'job.json').read_text());j.update(repairs=0,models=[self.config['models'][0]])
                        write_json(control/'job.json',j)
                        self.emit('control_start',variant=v['name'],policy='same first writer, no repair feedback')
                        t=time.monotonic()
                        try:
                            cs=self.build(control)
                            self.check_holdout(control,spec,v)
                            row['direct_control']={'accepted':True,'model_usd':cs['usd'],'seconds':time.monotonic()-t}
                        except Exception as e:
                            row['direct_control']={'accepted':False,'error':str(e)[:500],'seconds':time.monotonic()-t}
                        self.emit('control_done',variant=v['name'],**row['direct_control'])
                except Exception as e:
                    row.update(state='failed',error=(str(e)+' '+str(getattr(e,'stderr','')))[:1500])
                    self.emit('variant_failed',variant=v['name'],error=row['error'])
                self.save()
        except Exception as e:
            self.emit('experiment_failed',error=str(e)[:1000])
            self.save(error=str(e))
        finally:
            if not (self.out/'summary.json').exists():self.save()
            from .bundle import export_bundle
            self.emit('experiment_done',archive='reproducible-bundle.zip',results=self.results)
            try:
                export_bundle(self.out)
            finally:
                lock.close()
        return self.results

    def save(self,error=None):
        from utils.providers.budget import snapshot
        elapsed=time.time()-self.started
        rate=self.config['gpu_hourly_usd']
        for row in self.results:
            row['seed_summary']=aggregate_pairs(row.get('pairs',[]))
        write_json(self.out/'summary.json',{'schema_version':1,'mode':'live','results':self.results,
            'elapsed_s':elapsed,'budget':snapshot(self.ledger),'model_budget_usd':self.config['model_budget_usd'],
            'gpu_cost_estimate_usd':elapsed/3600*rate if rate is not None else None,
            'gpu_cost_scope':'experiment wall time only; rental setup/idle/storage excluded','error':error,
            'claim':'Measures implementation and experiment turnaround; does not establish architecture superiority.'})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--remote',default='bash -c');ap.add_argument('--remote-dir',default=str(ROOT))
    a=ap.parse_args()
    results=Experiment(json.loads(Path(a.config).read_text()),a.out,a.remote,a.remote_dir).run()
    if not results or any(r['state']!='complete' for r in results):
        raise SystemExit(1)

if __name__=='__main__':main()
