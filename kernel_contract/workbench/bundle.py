"""Portable source, frozen candidates, receipts and content hashes. Never include credentials."""
from pathlib import Path
import json
import shutil
import zipfile
from kernel_contract.evidence import sha256

ROOT = Path(__file__).resolve().parents[2]
REPRODUCE = r'''"""Verify all receipts, optionally rerun training with frozen kernels (no model calls)."""
import argparse, hashlib, json, os, pathlib, subprocess, sys
root=pathlib.Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--train',action='store_true',help='Run CUDA training from frozen kernels; no generation/API calls')
a=ap.parse_args()
manifest=json.loads((root/'manifest.json').read_text())
for name,digest in manifest['files'].items():
    path=root/name
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise SystemExit('Hash mismatch: '+name)
print('All bundle hashes match. This verifies integrity, not numerical correctness.')
if a.train:
    c=json.loads((root/'experiment/config.json').read_text())
    env={**os.environ,'PYTHONPATH':str(root/'source')}
    for i,v in enumerate(c['variants'],1):
        job=root/f'experiment/variant_{i}'
        if not (job/'kernel.py').exists() or not (job/'holdout_results.json').exists():continue
        if not json.loads((job/'holdout_results.json').read_text())['coverage']['all_cases_passed']:continue
        check=subprocess.run([sys.executable,'-m','muse_kernel.remote_eval',str(job),str(job/'kernel.py'),
                              '--grads','--holdout',str(job/'holdout_protocol.json')],
                             env=env,check=True,cwd=root/'source',capture_output=True,text=True)
        result=json.loads(next(line[7:] for line in check.stdout.splitlines() if line.startswith('RESULT ')))
        if not result['results'] or any(r['outcome']!='accept' for r in result['results']):
            raise SystemExit('Numerical recheck failed; training withheld: '+v['name'])
        for seed in c['seeds']:
            for mixer in ('checkpointed',str(job/'kernel.py')):
                label='reference' if mixer=='checkpointed' else 'kernel'
                cmd=[sys.executable,'-m','kernel_contract.train_demo','--layer',f'muse_jobs.j_{v["hash"]}',
                     '--mixer',mixer,'--seed',str(seed),'--out',str(root/f'reproduced/variant_{i}_{label}_{seed}.jsonl')]
                receipt=json.loads((job/f'{label}_{seed}.jsonl.meta.json').read_text())
                cmd+=['--expected-data-sha256',receipt['dataset_sha256']]
                if (root/'experiment/dataset.txt').exists():cmd+=['--data',str(root/'experiment/dataset.txt')]
                for key in ('steps','val_batches','batch','seq','heads','head_dim','width','layers'):
                    cmd+=['--'+key.replace('_','-'),str(c[key])]
                subprocess.run(cmd,env=env,check=True,cwd=root/'source')
'''


def export_bundle(experiment):
    experiment=Path(experiment)
    bundle=experiment/'bundle'
    bundle.mkdir(exist_ok=True)
    target=bundle/'experiment';target.mkdir(exist_ok=True)
    allowed={'.py','.json','.jsonl','.md'}
    for p in experiment.rglob('*'):
        if not p.is_file() or bundle in p.parents or '.source_snapshot' in p.parts or (p.suffix not in allowed and p.name!='dataset.txt') or p.name=='budget.json':continue
        if p.is_symlink():continue
        dest=target/p.relative_to(experiment);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    frozen=experiment/'.source_snapshot'
    if frozen.exists():shutil.copytree(frozen,bundle/'source',dirs_exist_ok=True)
    else:snapshot_source(bundle/'source')
    jobs=bundle/'source/muse_jobs';jobs.mkdir(exist_ok=True);(jobs/'__init__.py').write_text('')
    c=json.loads((experiment/'config.json').read_text())
    for i,v in enumerate(c['variants'],1):
        p=experiment/f'variant_{i}/reference.py'
        if p.exists():shutil.copy2(p,jobs/f'j_{v["hash"]}.py')
    (bundle/'reproduce.py').write_text(REPRODUCE)
    (bundle/'README.txt').write_text('Run python reproduce.py to verify file integrity.\nInstall source with pip install -e "./source[research]" on CUDA Linux.\nRun python reproduce.py --train to repeat paired training from the frozen kernels, without model calls.\nEnvironment versions and dataset hashes are in the receipts. Training reruns first repeat the frozen numerical checks and enforce the recorded dataset hash.\nFailures and unsupported cases are retained. No architecture-discovery claim is made.\n')
    files={str(p.relative_to(bundle)):sha256(p) for p in sorted(bundle.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    (bundle/'manifest.json').write_text(json.dumps({'schema_version':1,'source_snapshot_phase':'experiment_start' if frozen.exists() else 'export','files':files},indent=2))
    dest=experiment/'reproducible-bundle.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(bundle))
    return dest


def snapshot_source(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for package in ('kernel_contract','muse_kernel','utils','triton_kernel_agent','kernel_perf_agent','Fuser','scripts'):
        for p in (ROOT/package).rglob('*'):
            if p.suffix not in ('.py','.html','.j2','.yaml','.json') or '__pycache__' in p.parts or 'recur_cands' in p.parts:continue
            dest=destination/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for name in ('pyproject.toml','README.md','LICENSE'):
        shutil.copy2(ROOT/name,destination/name)
