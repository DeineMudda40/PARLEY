from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .distribution_codec import stick_break
from .model_spec import DecisionLayout
from .pctl_spec import ParsedPctl
from .runner import RunResult


def _decode_one(layout: DecisionLayout, x_row: np.ndarray) -> dict[str, float]:
    """
    Convert one internal decision vector x_row into a dict of
    parameter_name -> value, matching layout.parameter_columns.

    - int evolvables: rounded & clamped
    - double evolvables: clamped
    - distribution evolvables: expanded into name1..nameK via stick-breaking
    """
    x_row = np.asarray(x_row, dtype=float)
    params: dict[str, float] = {}

    for ev in layout.evolvables:
        start, length = layout.slices[ev.name]

        if ev.kind == "int":
            v = int(round(float(x_row[start])))
            v = max(int(ev.min_val), min(int(ev.max_val), v))
            params[ev.name] = v

        elif ev.kind == "double":
            v = float(x_row[start])
            v = max(ev.min_val, min(ev.max_val, v))
            params[ev.name] = v

        else:  # distribution
            u = x_row[start : start + length]
            probs = stick_break(u)
            k = probs.shape[0]
            for i in range(1, k + 1):
                params[f"{ev.name}{i}"] = float(probs[i - 1])

    return params


def save_parameters_tsv(path: str | Path, res: RunResult) -> None:
    """
    Save parameter table similar to seed file:

    - First row: header with parameter names (layout.parameter_columns)
    - Each row: decoded values per individual
    """
    layout = res.layout
    X = res.X

    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        # header
        f.write("\t".join(layout.parameter_columns) + "\n")

        for i in range(X.shape[0]):
            params = _decode_one(layout, X[i])
            row_vals = []
            for name in layout.parameter_columns:
                v = params[name]
                if float(v).is_integer():
                    row_vals.append(str(int(round(v))))
                else:
                    row_vals.append(f"{v:.10g}")
            f.write("\t".join(row_vals) + "\n")


def _restore_objective_values(F_min_space: np.ndarray, pctl: ParsedPctl) -> np.ndarray:
    """
    Our F is in minimization space (max objectives negated).
    Convert back to 'natural' values:
      - min objective: v
      - max objective: -v
    """
    F_min_space = np.asarray(F_min_space, dtype=float)
    F_nat = np.empty_like(F_min_space)

    for j, obj in enumerate(pctl.objectives):
        if obj.sense == "min":
            F_nat[:, j] = F_min_space[:, j]
        else:  # "max"
            F_nat[:, j] = -F_min_space[:, j]

    return F_nat


def save_front_tsv(path: str | Path, res: RunResult) -> None:
    """
    Save the objective values (Pareto front / final population) in natural sign:

    - First row: property text from PCTL
    - Rows: objective values (probabilities, costs, etc.)
    """
    path = Path(path)

    F_nat = _restore_objective_values(res.F, res.pctl)
    headers = [obj.prop for obj in res.pctl.objectives]

    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for i in range(F_nat.shape[0]):
            vals = [f"{F_nat[i, j]:.10g}" for j in range(F_nat.shape[1])]
            f.write("\t".join(vals) + "\n")

def save_hypervolume_history_tsv(path: str | Path, res: RunResult) -> None:
    """
    Save hypervolume history over time.

    Columns:
      - n_eval: number of evaluations so far
      - hypervolume: HV value at that point
      - time_sec: wall-clock time since start (seconds)
    """
    path = Path(path)

    evals = np.asarray(res.evals, dtype=int)
    hv_values = np.asarray(res.hv_values, dtype=float)
    timestamps = np.asarray(res.timestamps, dtype=float)

    assert evals.shape == hv_values.shape == timestamps.shape, (
        "evals, hv_values, and timestamps must have same length"
    )

    with path.open("w", encoding="utf-8") as f:
        f.write("n_eval\thypervolume\ttime_sec\n")
        for n_eval, hv, t in zip(evals, hv_values, timestamps):
            f.write(f"{n_eval}\t{hv:.10g}\t{t:.6f}\n")
