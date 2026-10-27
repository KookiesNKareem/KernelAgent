"""Atomic, persistent reservations shared by every model in an experiment."""
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import time
import uuid


class BudgetExceeded(RuntimeError):
    pass


@contextlib.contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix('.lock'), 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {'usd': 0.0, 'reservations': {}}
        data.setdefault('reservations', {})
        yield data
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)


def reserve(path, amount, cap):
    if not math.isfinite(amount) or not math.isfinite(cap) or amount < 0 or cap < 0:
        raise ValueError('budget amounts must be nonnegative')
    token = uuid.uuid4().hex
    with locked(path) as data:
        committed = data['usd'] + sum(r['usd'] for r in data['reservations'].values())
        if committed + amount > cap:
            raise BudgetExceeded(f'Budget ${cap:.2f}: committed ${committed:.2f}, next reservation ${amount:.2f}')
        data['reservations'][token] = {'usd': amount, 'created': time.time()}
    return token


def settle(path, token, actual):
    with locked(path) as data:
        reservation = data['reservations'].pop(token)
        data['usd'] += reservation['usd'] if actual is None else actual


def snapshot(path):
    with locked(path) as data:
        return {'usd': data['usd'], 'reserved_usd': sum(r['usd'] for r in data['reservations'].values())}
