from __future__ import annotations

from pathlib import Path
import numpy as np

from .model_spec import DecisionLayout
from .distribution_codec import inv_stick_break


def load_seed_table(path: str | Path, layout: DecisionLayout) -> np.ndarray:
    """
    Load an initial population from a whitespace-separated table.

    First row: header with parameter names (like layout.parameter_columns).
    Following rows: numeric values (for distributions: probabilities per bin).
    Returns X (n_individuals x n_var) in internal encoding (k-1 for distributions).
    """
    text = Path(path).read_text(encoding="utf-8").strip().splitlines()
    if not text:
        raise ValueError(f"Empty seed file: {path}")

    header = text[0].split()
    rows = [line.split() for line in text[1:] if line.strip()]
    if not rows:
        raise ValueError(f"No data rows in seed file: {path}")

    data = np.asarray(rows, dtype=float)
    col_idx = {name: i for i, name in enumerate(header)}

    X = np.zeros((data.shape[0], len(layout.xl)), dtype=float)

    for ev in layout.evolvables:
        start, length = layout.slices[ev.name]

        if ev.kind in ("int", "double"):
            col = col_idx[ev.name]
            X[:, start] = data[:, col]

        else:
            k = ev.bin_count
            # read probs name1..nameK
            probs = np.stack(
                [data[:, col_idx[f"{ev.name}{i}"]] for i in range(1, k + 1)],
                axis=1,
            )
            # convert each row p -> u (k-1)
            U = np.stack(
                [inv_stick_break(probs[i]) for i in range(probs.shape[0])], axis=0
            )
            if U.shape[1] != length:
                raise RuntimeError(
                    f"Internal length mismatch for distribution {ev.name}: "
                    f"expected {length}, got {U.shape[1]}"
                )
            X[:, start : start + length] = U

    return X
