"""Explicit empirical coverage and artifact identity; passing tests is not a proof."""
from collections import Counter
import hashlib
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coverage(results):
    counts = Counter(r['outcome'] for r in results)
    return {'total': len(results), **{k: counts[k] for k in ('accept', 'wrong', 'crash', 'refuse', 'skip')},
            'unsupported': [r['name'] for r in results if r['outcome'] == 'refuse'],
            'skipped': [r['name'] for r in results if r['outcome'] == 'skip'],
            'all_cases_passed': bool(results) and counts['accept'] == len(results)}


def eligible(results):
    """An observed success is mandatory. Explicitly unsupported size probes may be excluded."""
    return (any(r['name'] == 'observed' and r['outcome'] == 'accept' for r in results)
            and not any(r['outcome'] in ('wrong', 'crash') or
                        (r['outcome'] == 'refuse' and r.get('axis') != 'sizes') for r in results))
