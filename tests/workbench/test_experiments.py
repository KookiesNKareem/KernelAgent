import json
from pathlib import Path
import pytest
import torch
from kernel_contract.workbench.variants import DEFAULT_VARIANTS,variant,templates
from kernel_contract.workbench.experiment import validate
from kernel_contract.workbench.report import training_summary,paired_summary
from kernel_contract.layers._common import make_inputs


def test_structural_variants_change_computation_and_gradients():
    x=make_inputs(True,B=1,H=1,T=4,D=4,device='cpu',dtype=torch.float32)
    outputs=[]
    for item in DEFAULT_VARIANTS:
        v=variant(item);ns={};exec(v['source'],ns)
        args=[a.clone().requires_grad_() for a in x]
        out=ns['reference'](*args,.5);out.square().sum().backward()
        assert all(a.grad is not None and torch.isfinite(a.grad).all() for a in args)
        outputs.append(out.detach())
    assert all(not torch.allclose(outputs[0],o) for o in outputs[1:])


def test_source_edit_changes_identity_and_is_exported():
    v=variant(DEFAULT_VARIANTS[0]);edited=dict(DEFAULT_VARIANTS[0])
    edited['source']=edited['source'].replace('S = S *','S = 0.5 * S *')
    assert variant(edited)['hash']!=v['hash']
    with pytest.raises(ValueError):variant({**edited,'source':'import os\ndef reference(x): return x'})


def test_experiment_rejects_duplicates_and_bad_limits():
    with pytest.raises(ValueError):validate({'variants':[DEFAULT_VARIANTS[0]]*2})
    with pytest.raises(ValueError):validate({'model_budget_usd':float('nan')})
    with pytest.raises(ValueError):validate({'seeds':[0,0]})
    assert len(validate({})['variants'])==3


def test_reports_account_for_cold_start_and_failure(tmp_path):
    p=tmp_path/'train.jsonl'
    p.write_text('\n'.join(json.dumps({'step':i,'step_s':.2,'loss':3-i*.1,'elapsed_s':(i+1)*.2,'val_loss':3-i*.1,'peak_mem_gb':1}) for i in range(10)))
    k=training_summary(p,2.5)
    assert k['target_reached'] and k['time_to_target_s']==pytest.approx(1.2)
    r={**k,'median_step_ms':1000}
    pair=paired_summary(r,k,80)
    assert pair['step_speedup']==5
    assert pair['amortization_steps']==100
    assert pair['cold_time_to_target_s']==pytest.approx(81.2)
