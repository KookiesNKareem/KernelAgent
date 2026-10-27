import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pytest
import torch
from muse_kernel.identity import signature
from muse_kernel.infer import source_hash, standalone_module
from muse_kernel.recovery import train_step
from utils.providers.budget import reserve, settle, snapshot, BudgetExceeded
from kernel_contract.evidence import eligible, coverage
from kernel_contract.holdout import holdout_cases

GAIN=2.0

def reference(x):
    return x*GAIN


def test_dependency_capture_changes_hash(monkeypatch):
    before=source_hash(reference)
    namespace={};exec(standalone_module(reference),namespace)
    assert namespace['reference'](torch.tensor(2.)).item()==4
    monkeypatch.setitem(reference.__globals__,'GAIN',3.0)
    assert source_hash(reference)!=before


def test_signatures_pin_layout_offset_and_grad():
    x=torch.randn(4,4)
    assert signature((x,))!=signature((x.T,))
    assert signature((x,))!=signature((x.clone().requires_grad_(),))
    y=torch.ones(17)[1:].reshape(4,4)
    assert signature((x,))!=signature((y,))


def test_budget_concurrent_reservations(tmp_path):
    p=tmp_path/'budget.json'
    def attempt(_):
        try:return reserve(p,.6,1)
        except BudgetExceeded:return None
    with ThreadPoolExecutor(4) as pool:tokens=list(pool.map(attempt,range(4)))
    good=[x for x in tokens if x]
    assert len(good)==1
    assert snapshot(p)['reserved_usd']==.6
    settle(p,good[0],.2)
    assert snapshot(p)=={'usd':.2,'reserved_usd':0}
    reserve(p,.7,1)
    with pytest.raises(BudgetExceeded):reserve(p,.2,1)


def test_backward_recovery_preserves_optimizer_result():
    class Bad(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x):return x*2
        @staticmethod
        def backward(ctx,g):raise RuntimeError('deliberate backward failure')
    class Wrapper:
        disabled=False
        def __call__(self,x):return x*2 if self.disabled else Bad.apply(x)
        def disable(self,reason):self.disabled=True
    a=torch.nn.Linear(3,2);b=torch.nn.Linear(3,2);b.load_state_dict(a.state_dict())
    oa=torch.optim.Adam(a.parameters(),lr=.01);ob=torch.optim.Adam(b.parameters(),lr=.01)
    x=torch.randn(4,3);wrapped=Wrapper()
    loss,recovered=train_step(a,oa,lambda:wrapped(a(x)).square().mean(),[wrapped])
    train_step(b,ob,lambda:(b(x)*2).square().mean())
    assert recovered and wrapped.disabled
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q)


def test_coverage_does_not_call_refusals_or_skips_passes():
    rs=[{'name':'observed','outcome':'accept'}, {'name':'s2=129','axis':'sizes','outcome':'refuse'}, {'name':'k_ne_v','outcome':'skip'}]
    assert eligible(rs)
    assert not coverage(rs)['all_cases_passed']
    assert coverage(rs)['refuse']==1 and coverage(rs)['skip']==1
    assert not eligible([{'name':'observed','outcome':'skip'}])


def test_holdout_seeds_are_new_and_reproducible():
    a=holdout_cases(None,seed=12);b=holdout_cases(None,seed=12)
    assert a==b
    assert all(c['seed']>10000 and c['axis']=='holdout' for c in a['cases'])


def test_uninspectable_reference_stays_eager(monkeypatch,tmp_path):
    import muse_kernel
    monkeypatch.setenv('MUSE_KERNEL_CACHE',str(tmp_path))
    fn=eval('lambda x: x * 3')
    wrapped=muse_kernel.fast(fn)
    assert wrapped.disabled
    torch.testing.assert_close(wrapped(torch.tensor(2.)),torch.tensor(6.))
    assert not list(tmp_path.iterdir())


def test_incomplete_ready_cache_stays_eager(monkeypatch,tmp_path):
    import muse_kernel
    monkeypatch.setenv('MUSE_KERNEL_CACHE',str(tmp_path))
    wrapped=muse_kernel.fast(reference)
    wrapped.job.mkdir()
    (wrapped.job/'status.json').write_text(json.dumps({'state':'ready','kernel':'missing.py'}))
    wrapped._load_if_ready()
    assert wrapped.kernel is None
    torch.testing.assert_close(wrapped(torch.tensor(2.)),torch.tensor(4.))


def test_observed_contract_materializes_storage_offset():
    from kernel_contract.spec import capture,materialize
    x=torch.randn(30)[5:21].view(4,4).T
    spec=capture('torch.neg',(x,))
    args,_=materialize(spec,spec.sizes,0,'cpu')
    assert args[0].shape==x.shape
    assert args[0].stride()==x.stride()
    assert args[0].storage_offset()==5


def test_external_builder_captures_calls_without_starting_local_model(monkeypatch,tmp_path):
    import muse_kernel
    monkeypatch.setenv('MUSE_KERNEL_CACHE',str(tmp_path))
    monkeypatch.setenv('MUSE_KERNEL_EXTERNAL_BUILDER','1')
    def forbidden(*a,**k):raise AssertionError('Must not launch model client on the GPU worker')
    monkeypatch.setattr(muse_kernel.subprocess,'Popen',forbidden)
    wrapped=muse_kernel.fast(reference)
    for _ in range(2):wrapped(torch.randn(4,4,requires_grad=True))
    assert wrapped.status()['state']=='queued'
    job=json.loads((wrapped.job/'job.json').read_text())
    assert job['source']==standalone_module(reference)
    assert job['verified_signatures'][0][0]['requires_grad']


def test_cache_requiring_holdout_cannot_load_without_receipt(monkeypatch,tmp_path):
    import muse_kernel
    from kernel_contract.evidence import sha256
    monkeypatch.setenv('MUSE_KERNEL_CACHE',str(tmp_path))
    wrapped=muse_kernel.fast(reference)
    wrapped.job.mkdir()
    kernel=wrapped.job/'kernel.py';kernel.write_text('raise AssertionError("must not import")\n')
    (wrapped.job/'job.json').write_text(json.dumps({'require_holdout':True,'cache_version':2,'environment':wrapped.environment}))
    (wrapped.job/'status.json').write_text(json.dumps({'state':'ready','kernel':'kernel.py','kernel_sha256':sha256(kernel)}))
    wrapped._load_if_ready()
    assert wrapped.kernel is None
    assert wrapped.stats['cache missing fresh checks']==1
