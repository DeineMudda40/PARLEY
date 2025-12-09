from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .model_spec import DecisionLayout, Evolvable


# evolve int x [1..10];
# evolve double y [0.0..1.0];
# evolve distribution d [5];
_EVOLVE_RE = re.compile(
    r"^\s*evolve\s+(int|double|distribution)\s+([A-Za-z_]\w*)\s*\[([^\]]+)\]\s*;\s*$"
)


def _parse_range(spec: str) -> tuple[float, float]:
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*\.\.\s*([+-]?\d+(?:\.\d+)?)\s*$", spec)
    if not m:
        raise ValueError(f"Bad range spec in evolve: {spec!r} (expected like 1..10)")
    return float(m.group(1)), float(m.group(2))


def _parse_bin_count(spec: str) -> int:
    m = re.match(r"^\s*(\d+)\s*$", spec)
    if not m:
        raise ValueError(f"Bad bin_count spec in evolve distribution: {spec!r}")
    return int(m.group(1))


@dataclass(frozen=True)
class PreprocessedModel:
    preprocessed_path: Path
    layout: DecisionLayout


def preprocess_special_prism(
    input_path: os.PathLike | str,
    cache_dir: os.PathLike | str = ".evochecker_cache",
) -> PreprocessedModel:
    """
    Reads a 'special PRISM' file with `evolve` declarations,
    produces:
      * a standard PRISM file with consts (path in cache_dir)
      * a DecisionLayout describing the genotype
    Uses k-1 DOF for distributions (stick-breaking encoding).
    """
    input_path = Path(input_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    text = input_path.read_text(encoding="utf-8", errors="strict")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    out_path = cache_dir / f"{input_path.stem}.{digest}.prism"

    evolvables: list[Evolvable] = []
    kept_lines: list[str] = []
    const_lines: list[str] = []

    for line in text.splitlines(True):
        m = _EVOLVE_RE.match(line)
        if not m:
            kept_lines.append(line)
            continue

        kind, name, bracket = m.group(1), m.group(2), m.group(3)

        if kind == "int":
            lo, hi = _parse_range(bracket)
            evolvables.append(Evolvable("int", name, lo, hi))
            const_lines.append(f"const int {name};\n")

        elif kind == "double":
            lo, hi = _parse_range(bracket)
            evolvables.append(Evolvable("double", name, lo, hi))
            const_lines.append(f"const double {name};\n")

        else:  # distribution
            k = _parse_bin_count(bracket)
            # store bin_count in min_val/max_val
            evolvables.append(Evolvable("distribution", name, float(k), float(k)))
            # The model uses K constants: name1..nameK
            for i in range(1, k + 1):
                const_lines.append(f"const double {name}{i};\n")

    # insert const declarations after first line (dtmc/mdp/ctmc)
    if kept_lines:
        header = kept_lines[0]
        rest = kept_lines[1:]
        new_lines = [header] + const_lines + rest
    else:
        new_lines = const_lines

    # build DecisionLayout (x vector)
    xl: list[float] = []
    xu: list[float] = []
    slices: dict[str, tuple[int, int]] = {}
    columns: list[str] = []

    idx = 0
    for ev in evolvables:
        if ev.kind in ("int", "double"):
            slices[ev.name] = (idx, 1)
            xl.append(ev.min_val)
            xu.append(ev.max_val)
            columns.append(ev.name)
            idx += 1
        else:
            k = ev.bin_count
            # we use k-1 variables in [0,1] as stick-breaking parameters
            length = k - 1
            slices[ev.name] = (idx, length)
            xl.extend([0.0] * length)
            xu.extend([1.0] * length)
            # human-facing column names are actual bins
            for i in range(1, k + 1):
                columns.append(f"{ev.name}{i}")
            idx += length

    layout = DecisionLayout(
        evolvables=evolvables,
        xl=xl,
        xu=xu,
        slices=slices,
        parameter_columns=columns,
    )

    # one-time write of preprocessed prism file
    if not out_path.exists():
        out_path.write_text("".join(new_lines), encoding="utf-8")

    return PreprocessedModel(preprocessed_path=out_path, layout=layout)
