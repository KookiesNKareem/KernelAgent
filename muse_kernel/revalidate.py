"""Recheck a committed cache on the current CUDA environment without any model calls."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from kernel_contract.evidence import sha256, eligible, coverage
from .identity import environment


def revalidate(job):
    job=Path(job).resolve()
    meta=json.loads((job/'job.json').read_text())
    state=json.loads((job/'status.json').read_text())
    kernel=job/Path(state['kernel']).name
    if state.get('kernel_sha256') and sha256(kernel)!=state['kernel_sha256']:
        raise ValueError('Candidate hash changed; refusing to revalidate a modified cache')
    env=environment()
    if not env['gpu']:raise RuntimeError('Revalidation requires CUDA')
    # A temporary importable workspace avoids modifying installed site-packages.
    import tempfile,os
    with tempfile.TemporaryDirectory(prefix='muse_revalidate_') as tmp:
        root=Path(tmp);mods=root/'muse_jobs';mods.mkdir();(mods/'__init__.py').write_text('')
        shutil.copy2(job/'reference.py',mods/f'j_{meta["hash"]}.py')
        childenv={**os.environ,'PYTHONPATH':str(root)+os.pathsep+str(Path(__file__).resolve().parents[1])+os.pathsep+os.environ.get('PYTHONPATH','')}
        p=subprocess.run([sys.executable,'-m','muse_kernel.remote_eval',str(job),str(kernel),'--grads'],
                         env=childenv,capture_output=True,text=True,check=True,cwd=root)
        holdout_result = None
        if meta.get('require_holdout'):
            if not (job/'holdout_protocol.json').exists():
                raise ValueError('Required holdout protocol is missing')
            hp=subprocess.run([sys.executable,'-m','muse_kernel.remote_eval',str(job),str(kernel),'--grads',
                               '--holdout',str(job/'holdout_protocol.json')],
                              env=childenv,capture_output=True,text=True,check=True,cwd=root)
            holdout_result=json.loads(next(x[7:] for x in hp.stdout.splitlines() if x.startswith('RESULT ')))
            if not coverage(holdout_result['results'])['all_cases_passed']:
                raise ValueError('Cache failed holdout revalidation')
    result=json.loads(next(x[7:] for x in p.stdout.splitlines() if x.startswith('RESULT ')))
    if not eligible(result['results']):raise ValueError('Cache failed revalidation: '+json.dumps(coverage(result['results'])))
    meta.update(environment=env,cache_version=2)
    # Legacy repr signatures are intentionally not silently migrated.
    if not isinstance(meta.get('verified_signatures'),list) or any(isinstance(s,str) for s in meta['verified_signatures']):
        raise ValueError('Legacy signature cache: rebuild from observed calls')
    state.update(kernel_sha256=sha256(kernel),coverage=coverage(result['results']),state='ready',holdout_passed=holdout_result is not None)
    (job/'revalidation.json').write_text(json.dumps({'environment':env,'result':result,'holdout_result':holdout_result},indent=2))
    (job/'job.json').write_text(json.dumps(meta,indent=2))
    (job/'status.json').write_text(json.dumps(state,indent=2))
    return state


def main():
    ap=argparse.ArgumentParser();ap.add_argument('job');a=ap.parse_args()
    print(json.dumps(revalidate(a.job),indent=2))

if __name__=='__main__':main()
