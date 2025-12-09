from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .model_spec import DecisionLayout
from .pctl_spec import ParsedPctl, ObjectiveSpec, ConstraintSpec
from .distribution_codec import stick_break

# Global context (per worker process)
_CTX: "_WorkerContext | None" = None


@dataclass
class _WorkerContext:
    layout: DecisionLayout
    pctl: ParsedPctl
    stormpy: Any
    symbolic_desc: Any
    properties: list[Any]  # objectives first, then constraints
    n_obj: int
    n_constr: int


def _decode_to_constants(layout: DecisionLayout, x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    out: dict[str, float] = {}

    for ev in layout.evolvables:
        start, length = layout.slices[ev.name]

        if ev.kind == "int":
            v = int(round(float(x[start])))
            v = max(int(ev.min_val), min(int(ev.max_val), v))
            out[ev.name] = v

        elif ev.kind == "double":
            v = float(x[start])
            v = max(ev.min_val, min(ev.max_val, v))
            out[ev.name] = v

        else:
            # distribution: stick-breaking from k-1 values in [0,1]
            u = x[start : start + length]
            probs = stick_break(u)
            k = probs.shape[0]
            for i in range(1, k + 1):
                out[f"{ev.name}{i}"] = float(probs[i - 1])

    return out


def _constants_string(consts: dict[str, float]) -> str:
    parts = []
    for k, v in consts.items():
        if float(v).is_integer():
            parts.append(f"{k}={int(v)}")
        else:
            parts.append(f"{k}={v}")
    return ",".join(parts)


def init_worker(
    preprocessed_prism_path: str,
    layout: DecisionLayout,
    pctl: ParsedPctl,
) -> None:
    """
    Called once per worker process by multiprocessing.Pool(initializer=...).
    """
    global _CTX
    import stormpy  # local import inside worker

    prism_program = stormpy.parse_prism_program(preprocessed_prism_path)

    # symbolic description so we can instantiate constants per individual
    symbolic_desc = stormpy.SymbolicModelDescription(prism_program)

    # Parse each property separately: objectives first, then constraints.
    properties: list[Any] = []

    for o in pctl.objectives:
        props = stormpy.parse_properties_for_prism_program(o.prop, prism_program)
        if len(props) != 1:
            raise RuntimeError(
                f"Expected 1 property for objective, got {len(props)}: {o.prop}"
            )
        properties.append(props[0])

    for c in pctl.constraints:
        props = stormpy.parse_properties_for_prism_program(c.prop, prism_program)
        if len(props) != 1:
            raise RuntimeError(
                f"Expected 1 property for constraint, got {len(props)}: {c.prop}"
            )
        properties.append(props[0])

    _CTX = _WorkerContext(
        layout=layout,
        pctl=pctl,
        stormpy=stormpy,
        symbolic_desc=symbolic_desc,
        properties=properties,
        n_obj=len(pctl.objectives),
        n_constr=len(pctl.constraints),
    )

    _CTX = _WorkerContext(
        layout=layout,
        pctl=pctl,
        stormpy=stormpy,
        symbolic_desc=symbolic_desc,
        properties=properties,
        n_obj=len(pctl.objectives),
        n_constr=len(pctl.constraints),
    )


def _model_check(stormpy, model, prop):
    # Robust against version differences
    if hasattr(stormpy, "model_checking"):
        return stormpy.model_checking(model, prop)
    if hasattr(stormpy, "check_model_sparse"):
        return stormpy.check_model_sparse(model, prop)
    if hasattr(stormpy, "check_model_dd"):
        return stormpy.check_model_dd(model, prop)
    raise AttributeError("No suitable model checking function found in stormpy.")


def evaluate_one(x_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a single individual.

    Returns:
      F: objectives array (minimization; max objectives are negated)
      G: constraint violations (>=0, 0 means satisfied)
    """
    global _CTX
    if _CTX is None:
        raise RuntimeError("Worker not initialized (init_worker not called).")

    stormpy = _CTX.stormpy

    consts = _decode_to_constants(_CTX.layout, x_row)
    const_str = _constants_string(consts)

    const_def = _CTX.symbolic_desc.parse_constant_definitions(const_str)
        # inst_desc is a SymbolicModelDescription with constants instantiated
    inst_desc = _CTX.symbolic_desc.instantiate_constants(const_def)

    # NEW: pass a PrismProgram to build_sparse_model
    prism_program_inst = inst_desc.as_prism_program()
    model = stormpy.build_sparse_model(prism_program_inst, _CTX.properties)


    # initial state index
    init_state = 0
    try:
        bv = model.initial_states
        if hasattr(bv, "get_next_set_index"):
            idx = bv.get_next_set_index(0)
            if idx != -1:
                init_state = int(idx)
    except Exception:
        init_state = 0

    # compute values for all properties (objectives + constraints)
    values: list[float] = []

    for prop in _CTX.properties:
        r = _model_check(stormpy, model, prop)
        if hasattr(r, "at"):
            values.append(float(r.at(init_state)))
        else:
            # boolean result fallback
            if hasattr(r, "get_truth_values"):
                tv = r.get_truth_values()
                val = 1.0 if tv.get(init_state) else 0.0
            else:
                val = 1.0 if bool(r) else 0.0
            values.append(val)

    values = np.asarray(values, dtype=float)

    # build F (objectives)
    F = np.empty(_CTX.n_obj, dtype=float)
    for j, obj in enumerate(_CTX.pctl.objectives):
        v = values[j]
        if not np.isfinite(v):
            v = 1e9
        # pymoo minimizes; if sense == "max", we negate
        F[j] = v if obj.sense == "min" else -v

    # build G (constraints) as violations >= 0
    G = np.empty(_CTX.n_constr, dtype=float)
    offset = _CTX.n_obj
    for k, con in enumerate(_CTX.pctl.constraints):
        v = values[offset + k]
        if not np.isfinite(v):
            v = 1e9

        if con.cmp == "<=":
            # want v <= bound; violation = max(0, v - bound)
            viol = max(0.0, v - con.bound)
        else:
            # want v >= bound; violation = max(0, bound - v)
            viol = max(0.0, con.bound - v)

        G[k] = viol

    return F, G
