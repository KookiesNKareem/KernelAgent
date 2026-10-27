"""Comparable training summaries; preserve failures and unknown costs."""
import json
import statistics
from pathlib import Path


def training_summary(path, target=None):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    steps = [r for r in rows if 'step_s' in r]
    elapsed=0.0
    for r in steps:
        elapsed += r['step_s']
        r.setdefault('elapsed_s',elapsed)
    vals = [r for r in steps if 'val_loss' in r]
    warm = steps[5:] or steps
    hit = next((r for r in vals if target is not None and r['val_loss'] <= target), None)
    return {'steps': len(steps), 'median_step_ms': statistics.median(r['step_s'] for r in warm)*1000 if warm else None,
            'final_val_loss': vals[-1]['val_loss'] if vals else None,
            'peak_mem_gb': max((r['peak_mem_gb'] for r in vals if r.get('peak_mem_gb') is not None), default=None),
            'time_to_target_s': hit.get('elapsed_s') if hit else None,
            'target_reached': hit is not None, 'elapsed_s': steps[-1].get('elapsed_s') if steps else None,
            'out_of_memory': any(r.get('out_of_memory') for r in rows),
            'curve': [{k:r[k] for k in ('step','loss','val_loss','elapsed_s') if k in r} for r in steps]}


def paired_summary(reference, kernel, build_s):
    r, k = reference.get('median_step_ms'), kernel.get('median_step_ms')
    saved_s = (r-k)/1000 if r is not None and k is not None else None
    return {'step_speedup': r/k if r and k else None,
            'amortization_steps': build_s/saved_s if saved_s and saved_s > 0 else None,
            'cold_time_to_target_s': build_s + kernel['time_to_target_s'] if kernel.get('time_to_target_s') is not None else None,
            'cached_time_to_target_s': kernel.get('time_to_target_s'),
            'cold_training_wall_s': build_s + kernel['elapsed_s'] if kernel.get('elapsed_s') is not None else None}


def aggregate_pairs(pairs):
    """Descriptive seed summaries, without significance or architecture claims."""
    def summarize(values):
        values=[v for v in values if v is not None]
        return {'n':len(values),'mean':statistics.mean(values) if values else None,
                'sd':statistics.stdev(values) if len(values)>1 else None}
    return {m:{key:summarize([p.get(m,{}).get(key) for p in pairs])
               for key in ('final_val_loss','median_step_ms','peak_mem_gb','elapsed_s')}
            for m in ('reference','kernel')}
