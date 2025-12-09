from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


EvolvableKind = Literal["int", "double", "distribution"]


@dataclass(frozen=True)
class Evolvable:
    kind: EvolvableKind
    name: str
    # for int/double: [min_val, max_val]
    # for distribution: min_val == max_val == bin_count (stored as float)
    min_val: float
    max_val: float

    @property
    def is_distribution(self) -> bool:
        return self.kind == "distribution"

    @property
    def bin_count(self) -> int:
        if self.kind != "distribution":
            raise ValueError("Not a distribution evolvable.")
        return int(self.min_val)


@dataclass(frozen=True)
class DecisionLayout:
    """Mapping between flat decision vector x and logical parameters."""
    evolvables: Sequence[Evolvable]
    xl: list[float]
    xu: list[float]
    # slices[name] = (start_index_in_x, length)
    slices: dict[str, tuple[int, int]]
    # column names when saving/reading human-readable parameter tables
    parameter_columns: list[str]
