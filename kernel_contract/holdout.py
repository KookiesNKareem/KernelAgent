"""Fresh post-selection cases; never sent to the writer for repair."""
import secrets


def holdout_cases(spec, seed=None, count=12):
    import random
    seed = secrets.randbits(31) if seed is None else seed
    rng = random.Random(seed)
    cases = []
    layouts = ('contiguous', 'strided2', 'strided3_off1', 'transposed', 'leading_swap')
    stresses = ('gate_typical', 'gate_mid', 'gate_high', 'decay_strong', 'decay_none', 'gate_edges')
    for i in range(count):
        cases.append({'name': f'holdout_{i}', 'axis': 'holdout', 'seed': rng.randrange(10000, 2**31),
                      'layout': layouts[i % len(layouts)], 'stress': stresses[i % len(stresses)]})
    return {'seed': seed, 'policy': 'sampled after candidate selection; no repair feedback', 'cases': cases}
