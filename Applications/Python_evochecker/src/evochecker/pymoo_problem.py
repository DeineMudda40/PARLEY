from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np
from pymoo.core.problem import Problem

from .model_spec import DecisionLayout
from .pctl_spec import ParsedPctl
from .storm_eval import evaluate_one


class ParallelStormProblem(Problem):
    def __init__(
        self,
        *,
        layout: DecisionLayout,
        pctl: ParsedPctl,
        preprocessed_prism_path: str,
        pool: mp.pool.Pool,
        n_var: int,
        n_obj: int,
        n_ieq_constr: int,
        xl: np.ndarray,
        xu: np.ndarray,
    ):
        """
        layout, pctl, preprocessed_prism_path, pool are stored as attributes,
        and Problem.__init__ is called with the usual arguments.
        """
        self.layout = layout
        self.pctl = pctl
        self.preprocessed_prism_path = preprocessed_prism_path
        self.pool = pool

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=n_ieq_constr,
            xl=xl,
            xu=xu,
        )

    @staticmethod
    def create(
        *,
        layout: DecisionLayout,
        pctl: ParsedPctl,
        preprocessed_prism_path: str,
        pool: mp.pool.Pool,
    ) -> "ParallelStormProblem":
        n_obj = len(pctl.objectives)
        n_constr = len(pctl.constraints)

        return ParallelStormProblem(
            layout=layout,
            pctl=pctl,
            preprocessed_prism_path=preprocessed_prism_path,
            pool=pool,
            n_var=len(layout.xl),
            n_obj=n_obj,
            n_ieq_constr=n_constr,
            xl=np.asarray(layout.xl, dtype=float),
            xu=np.asarray(layout.xu, dtype=float),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X, dtype=float)

        # Parallel map over individuals
        results = self.pool.map(evaluate_one, [row for row in X])

        if self.n_obj > 0:
            F = np.vstack([r[0] for r in results])
        else:
            F = np.zeros((len(X), 0), dtype=float)

        if self.n_ieq_constr > 0:
            G = np.vstack([r[1] for r in results])
        else:
            G = np.zeros((len(X), 0), dtype=float)

        out["F"] = F
        out["G"] = G
