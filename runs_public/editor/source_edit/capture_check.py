"""CUDA capture-only check; pass the cache parent containing the original reference cache."""
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ['MUSE_KERNEL_EXTERNAL_BUILDER']='1'
os.environ['MUSE_KERNEL_CACHE']=sys.argv[1]
import torch
from kernel_contract.train_demo import make_lm, loss_of

original='80d4dce92143eb856dc192a9'
assert (Path(sys.argv[1])/('reference_'+original)).is_dir()
spec=importlib.util.spec_from_file_location('edited_research',Path(__file__).with_name('script.py'))
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
torch.manual_seed(0)
model=make_lm('edited_research',m.reference,device='cuda',d=64,layers=1,heads=2,head_dim=16)
for _ in range(2):
    model.zero_grad()
    loss_of(model,torch.randint(0,256,(2,65),device='cuda')).backward()
assert m.reference.hash!=original
assert m.reference.status()['calls'].get('kernel',0)==0
assert m.reference.status()['calls']['reference']==2
assert m.reference.status()['state']=='queued'
print(json.dumps({'old_hash':original,'edited_hash':m.reference.hash,
                  'calls':m.reference.status()['calls'],'state':m.reference.status()['state']}))
