"""Structural recurrence edits. The exact PyTorch source, not a coefficient label, defines a variant.

References are trusted local researcher code executed on the configured worker. AST checks validate
our training adapter interface; they are not a security sandbox.
"""
import ast
import hashlib
from pathlib import Path

LAYERS=Path(__file__).resolve().parents[1]/'layers'
SIGNATURE=['q','k','v','e','g','beta','gamma','scale']
TEMPLATES=[
    ('Erase then write','b04_eda_channel','Erase key-addressed memory before the delta write.'),
    ('Write then erase','b05_write_erase','Reverse the noncommuting operations; the new write is erased too.'),
    ('Value-side erase','b06_value_erase','Erase in value space rather than key space; the state is transformed on both sides.'),
    ('Read before write','b08_read_before','Read the decayed previous state, preventing the current write from entering the current output.'),
]


def template(name,module,description):
    # Extract only the reference, without importing or executing the module.
    path=LAYERS/(module+'.py')
    if not path.exists() and module=='b04_eda_channel':path=LAYERS/'b04_eda_per_channel.py'
    tree=ast.parse(path.read_text())
    ref=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reference')
    doc=ast.get_docstring(tree) or ''
    equation=' '.join(line.strip() for line in doc.splitlines()[1:]).split(',    o_t')[0]
    source='import torch\nCHANNEL_DECAY = True\n\n'+ast.unparse(ref)+'\n'
    return {'name':name,'source':source,'equation':equation,'description':description,'template':module}


def templates():
    return [template(*t) for t in TEMPLATES]


def variant(config):
    name=str(config.get('name','Untitled variant')).strip()
    if not name or len(name)>60:raise ValueError('Name must have 1–60 characters')
    source=config.get('source')
    if not isinstance(source,str) or not 20<len(source)<24000:raise ValueError('Provide a PyTorch reference module (under 24 KB)')
    tree=ast.parse(source)
    refs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reference']
    if len(refs)!=1:raise ValueError('Define exactly one reference function')
    ref=refs[0]
    if [a.arg for a in ref.args.args]!=SIGNATURE or ref.args.vararg or ref.args.kwarg or ref.args.kwonlyargs or ref.decorator_list:
        raise ValueError('This training adapter requires reference(q, k, v, e, g, beta, gamma, scale), without decorators')
    for node in tree.body:
        if isinstance(node,ast.Import):
            if any(a.name!='torch' for a in node.names):raise ValueError('Reference modules may import torch only; inline math helpers')
        elif isinstance(node,ast.ImportFrom):
            if node.module!='__future__':raise ValueError('Inline helpers in the reference module')
        elif isinstance(node,(ast.FunctionDef,ast.Assign)):
            if isinstance(node,ast.Assign):
                if not all(isinstance(t,ast.Name) and t.id=='CHANNEL_DECAY' for t in node.targets) or not isinstance(node.value,ast.Constant) or node.value.value is not True:
                    raise ValueError('This adapter uses CHANNEL_DECAY = True; put other constants inside the function')
        elif not (isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str)):
            raise ValueError('Top-level reference code may contain torch imports, functions and CHANNEL_DECAY only')
    canonical=ast.unparse(tree)+'\n'
    if not any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHANNEL_DECAY' for t in n.targets) for n in tree.body):
        canonical='CHANNEL_DECAY = True\n'+canonical
    digest=hashlib.sha256(canonical.encode()).hexdigest()[:24]
    return {'name':name,'source':canonical,'hash':digest,'equation':str(config.get('equation','Researcher-defined PyTorch recurrence'))[:1500],
            'description':str(config.get('description','Structural edit supplied by the researcher'))[:1500]}


DEFAULT_VARIANTS=templates()[:3]
