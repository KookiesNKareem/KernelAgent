"""Cases a kernel must pass to honor the contract of the op it replaces.

Axes: seed (redraw the inputs), values (transform every floating input), sizes (perturb each
size symbol), layout (same logical values through a different memory layout). The dense
contract is seed + values + sizes at the observed layout; the strided contract adds layout.
"""

from __future__ import annotations

from .spec import OpSpec

VALUE_TRANSFORMS = {
    "scale_up": lambda x: x * 1e3,
    "scale_down": lambda x: x * 1e-3,
    "shift": lambda x: x + 100.0,
    "negative": lambda x: -x.abs() - 1.0,
    "half_zero": lambda x: x.clamp(min=0),
    "constant": lambda x: x * 0 + 0.5,
    "scale_mid": lambda x: x * 37.0,
    "shift_neg": lambda x: x - 100.0,
}

LAYOUTS = ("contiguous", "strided2", "strided3_off1", "transposed", "leading_swap")


def build_cases(spec: OpSpec, contract: str = "strided") -> list[dict]:
    cases: list[dict] = [{"name": "observed", "axis": "seed", "seed": 0}]
    cases += [{"name": f"seed{s}", "axis": "seed", "seed": s} for s in (1, 2)]
    cases += [{"name": t, "axis": "values", "seed": 0, "transform": t} for t in VALUE_TRANSFORMS]
    for s, v in spec.sizes.items():
        for nv in sorted({v + 1, v // 2 + 1} - {v}):
            cases.append({"name": f"{s}={nv}", "axis": "sizes", "seed": 0, "sizes": {s: nv}})
    domains = {a["tensor"].get("domain") for a in spec.args if isinstance(a, dict) and "tensor" in a}
    if "logdecay" in domains:
        cases += [{"name": n, "axis": "values", "seed": 0, "stress": n} for n in ("decay_none", "decay_strong")]
    if "unit" in domains:
        cases.append({"name": "gate_edges", "axis": "values", "seed": 0, "stress": "gate_edges"})
    if contract == "strided":
        cases += [{"name": lay, "axis": "layout", "seed": 0, "layout": lay} for lay in LAYOUTS]
    return cases
