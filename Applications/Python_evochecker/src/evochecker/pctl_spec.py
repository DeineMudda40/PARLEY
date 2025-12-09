from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import re

Sense = Literal["min", "max"]
Cmp = Literal["<=", ">="]

_OBJ_RE = re.compile(r"(.*)\s+@objective\s+(min|max)\s*$")
_CON_RE = re.compile(r"(.*)\s+@constraint\s*(<=|>=)\s*([+-]?\d+(?:\.\d+)?)\s*$")


@dataclass(frozen=True)
class ObjectiveSpec:
    prop: str     # pure PRISM property
    sense: Sense  # "min" or "max"


@dataclass(frozen=True)
class ConstraintSpec:
    prop: str     # property that yields a numeric value
    cmp: Cmp      # "<=" or ">="
    bound: float  # threshold


@dataclass(frozen=True)
class ParsedPctl:
    objectives: list[ObjectiveSpec]
    constraints: list[ConstraintSpec]
    objective_headers: list[str]
    constraint_headers: list[str]


def parse_pctl_with_tags(pctl_path: str | Path) -> ParsedPctl:
    objectives: list[ObjectiveSpec] = []
    constraints: list[ConstraintSpec] = []
    obj_hdr: list[str] = []
    con_hdr: list[str] = []

    for raw in Path(pctl_path).read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("//") or s.startswith("#"):
            continue

        m = _OBJ_RE.match(s)
        if m:
            prop = m.group(1).strip()
            sense = m.group(2)  # type: ignore
            objectives.append(ObjectiveSpec(prop=prop, sense=sense))
            obj_hdr.append(f"{prop}\t({sense})")
            continue

        m = _CON_RE.match(s)
        if m:
            prop = m.group(1).strip()
            cmp = m.group(2)  # type: ignore
            bound = float(m.group(3))
            constraints.append(ConstraintSpec(prop=prop, cmp=cmp, bound=bound))
            con_hdr.append(f"{prop}\t({cmp} {bound})")
            continue

        raise ValueError(
            "Every property line must end with either "
            "'@objective min|max' or '@constraint <=|>= number'. "
            f"Bad line: {s}"
        )

    if not objectives:
        raise ValueError("Need at least one @objective line in the PCTL file.")

    return ParsedPctl(
        objectives=objectives,
        constraints=constraints,
        objective_headers=obj_hdr,
        constraint_headers=con_hdr,
    )
