from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling

from .special_prism import preprocess_special_prism
from .pctl_spec import parse_pctl_with_tags, ParsedPctl
from .pymoo_problem import ParallelStormProblem
from .storm_eval import init_worker
from .sampling import SeededSampling
from .seed_loader import load_seed_table
from .model_spec import DecisionLayout


from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
import numpy as np
import time


class HypervolumeCallback(Callback):
    def __init__(self, ref_point):
        super().__init__()
        self.hv = HV(ref_point=ref_point)
        
        self.data["evals"] = []
        self.data["hv"] = []
        self.data["timestamps"] = []
        
        self._start_time = time.perf_counter()

    def notify(self, algorithm):
        now = time.perf_counter()
        self.data["timestamps"].append(now - self._start_time)
        
        F = algorithm.opt.get("F")

        if F is not None and len(F) > 0:
            hv_value = self.hv(F)
        else:
            hv_value = 0.0

        self.data["evals"].append(algorithm.evaluator.n_eval)
        self.data["hv"].append(hv_value)


@dataclass(frozen=True)
class RunResult:
    X: np.ndarray  # decision vectors in internal encoding
    F: np.ndarray  # objective values (minimization, max-objectives negated)
    G: Optional[np.ndarray]
    layout: DecisionLayout
    pctl: ParsedPctl
    evals: np.ndarray
    hv_values: np.ndarray
    timestamps: np.ndarray


def run_nsga2(
    *,
    special_prism_path: str,
    pctl_path: str,
    population_size: int = 50,
    max_evaluations: int = 500,
    n_workers: Optional[int] = None,
    rng_seed: int = 1,
    initial_population_path: Optional[str] = None,
    ref_point=np.array([1.0, 100.0]),
) -> RunResult:
    """
    Main entrypoint.

    special_prism_path: PRISM model with 'evolve' declarations
    pctl_path: PCTL with '@objective min|max' and '@constraint' tags
    population_size: NSGA-II population size
    max_evaluations: total number of evaluations
    n_workers: processes for parallel model checking; default = cpu_count-1
    rng_seed: seed for pymoo / numpy RNG (not the GA seed file!)
    initial_population_path: path to seed parameter table (optional)
    """
    pre = preprocess_special_prism(special_prism_path)
    pctl: ParsedPctl = parse_pctl_with_tags(pctl_path)

    n_workers = n_workers or max(1, mp.cpu_count() - 1)
    ctx = mp.get_context("spawn")

    pool = ctx.Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(str(pre.preprocessed_path), pre.layout, pctl),
    )

    try:
        # initial population (optional)
        if initial_population_path is not None:
            X_seed = load_seed_table(initial_population_path, pre.layout)
        else:
            X_seed = None

        problem = ParallelStormProblem.create(
            layout=pre.layout,
            pctl=pctl,
            preprocessed_prism_path=str(pre.preprocessed_path),
            pool=pool,
        )

        # Only use SeededSampling if a seed population is actually provided.
        if X_seed is None or X_seed.shape[0] == 0:
            # No seed → use NSGA2's default random initialization
            algorithm = NSGA2(
                pop_size=population_size,
            )
        else:
            from .sampling import SeededSampling

            algorithm = NSGA2(
                pop_size=population_size,
                sampling=SeededSampling(pre.layout, X_seed),
                eliminate_duplicates=True,
            )

        termination = get_termination("n_eval", max_evaluations)

        hv_callback = HypervolumeCallback(ref_point)

        res = minimize(
            problem,
            algorithm,
            termination=termination,
            seed=rng_seed,
            verbose=True,
            save_history=False,
            callback=hv_callback,
        )

        X = np.asarray(res.X)
        F = np.asarray(res.F)
        G = np.asarray(res.G) if getattr(res, "G", None) is not None else None

        return RunResult(
            X=X,
            F=F,
            G=G,
            layout=pre.layout,
            pctl=pctl,
            evals=np.array(hv_callback.data["evals"]),
            hv_values=np.array(hv_callback.data["hv"]),
            timestamps=np.array(hv_callback.data["timestamps"]),
        )

    finally:
        pool.close()
        pool.join()
