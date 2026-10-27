import json
import threading
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
from http.server import ThreadingHTTPServer
import pytest
from kernel_contract.workbench.server import make_handler
from kernel_contract.workbench.bundle import export_bundle
from kernel_contract.workbench.variants import DEFAULT_VARIANTS
from kernel_contract.workbench.experiment import validate


def test_local_http_preview_and_csrf(tmp_path):
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(tmp_path,'bash -c','/tmp','test-token'))
    t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        assert urllib.request.urlopen(base).status==200
        data=json.dumps({'variants':DEFAULT_VARIANTS[:2]}).encode()
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(base+'/api/preview',data=data))
        assert e.value.code==403
        req=urllib.request.Request(base+'/api/preview',data=data,headers={'X-Workbench-Token':'test-token'})
        result=json.load(urllib.request.urlopen(req))
        assert len(result['variants'])==2 and result['variants'][1]['diff']
        req=urllib.request.Request(base+'/api/run',data=data,headers={'X-Workbench-Token':'test-token','Origin':'https://unrelated.example'})
        with pytest.raises(urllib.error.HTTPError) as e:urllib.request.urlopen(req)
        assert e.value.code==403
    finally:server.shutdown();server.server_close();t.join()


def test_bundle_integrity_and_no_secrets(tmp_path):
    c=validate({'variants':DEFAULT_VARIANTS[:1]})
    (tmp_path/'config.json').write_text(json.dumps(c))
    (tmp_path/'.env').write_text('MODEL_API_KEY=do-not-export')
    (tmp_path/'budget.json').write_text('{}')
    (tmp_path/'variant_1').mkdir()
    (tmp_path/'variant_1/reference.py').write_text(c['variants'][0]['source'])
    archive=export_bundle(tmp_path)
    with zipfile.ZipFile(archive) as z:
        assert 'reproduce.py' in z.namelist()
        assert not any('.env' in n or n.endswith('budget.json') for n in z.namelist())
        manifest=json.loads(z.read('manifest.json'))
        import hashlib
        for name,h in manifest['files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h
        assert any(n.startswith('source/muse_jobs/j_') for n in z.namelist())


def test_replay_withholds_training_when_current_gpu_checks_fail(tmp_path,monkeypatch):
    """A prior passing receipt must not bypass checks in the replay environment."""
    import subprocess,sys
    from kernel_contract.workbench.bundle import REPRODUCE
    config=validate({'variants':DEFAULT_VARIANTS[:1]})
    (tmp_path/'config.json').write_text(json.dumps(config))
    job=tmp_path/'variant_1';job.mkdir()
    (job/'reference.py').write_text(config['variants'][0]['source'])
    (job/'kernel.py').write_text('def kernel_function(*args): return None\n')
    (job/'holdout_results.json').write_text(json.dumps({'coverage':{'all_cases_passed':True}}))
    (job/'holdout_protocol.json').write_text('{}')
    export_bundle(tmp_path)
    calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd,0,stdout='RESULT '+json.dumps({'results':[{'outcome':'wrong'}]}))
    monkeypatch.setattr(subprocess,'run',run)
    monkeypatch.setattr(sys,'argv',['reproduce.py','--train'])
    with pytest.raises(SystemExit,match='Numerical recheck failed'):
        exec(REPRODUCE,{'__file__':str(tmp_path/'bundle/reproduce.py'),'__name__':'__main__'})
    assert len(calls)==1 and 'muse_kernel.remote_eval' in calls[0]


def test_busy_worker_exports_failure_without_running_gpu(tmp_path,monkeypatch):
    import fcntl,hashlib,tempfile
    from kernel_contract.workbench.experiment import Experiment
    remote_dir=str(tmp_path)
    key=hashlib.sha256(('bash -c '+remote_dir).encode()).hexdigest()[:20]
    with open(Path(tempfile.gettempdir())/('recurrence-worker-'+key+'.lock'),'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        exp=Experiment({'variants':DEFAULT_VARIANTS[:1]},tmp_path/'busy','bash -c',remote_dir)
        def forbidden(*a,**k):raise AssertionError('Busy worker must not be touched')
        monkeypatch.setattr(exp,'remote_run',forbidden)
        assert exp.run()==[]
    assert 'Another experiment' in json.loads((exp.out/'summary.json').read_text())['error']
    assert (exp.out/'reproducible-bundle.zip').is_file()
