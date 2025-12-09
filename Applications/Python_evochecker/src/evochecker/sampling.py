from __future__ import annotations

import numpy as np
from pymoo.core.sampling import Sampling

from .model_spec import DecisionLayout


class SeededSampling(Sampling):
    """
    Initial population sampling:

    - If X_seed is None or empty: we let pymoo failover to its own default
      sampling by returning None (but we avoid that now by only using this
      class when a seed is present).
    - If X_seed has >= pop_size rows: use the first pop_size rows.
    - If X_seed has < pop_size rows:
        * place all seed individuals once
        * fill the remaining rows with new random individuals
          (no repetition of seed rows).
    """

    def __init__(self, layout: DecisionLayout, X_seed: np.ndarray | None):
        super().__init__()
        self.layout = layout
        self.X_seed = None if X_seed is None else np.asarray(X_seed, dtype=float)

    def _do(self, problem, n_samples, **kwargs):
        # If somehow called with no seed, just let pymoo handle it
        if self.X_seed is None or self.X_seed.shape[0] == 0:
            return None

        X_seed = self.X_seed
        n_var = problem.n_var

        if X_seed.shape[1] != n_var:
            raise ValueError(
                f"Seed population has {X_seed.shape[1]} variables, "
                f"but problem expects {n_var}."
            )

        # number of seed individuals we actually use
        s = min(X_seed.shape[0], n_samples)

        # Full initial population
        X = np.empty((n_samples, n_var), dtype=float)

        # 1) Put seed individuals (each at most once)
        X[:s, :] = X_seed[:s, :]

        # 2) Fill the rest with random individuals (no repetition of seeds)
        if s < n_samples:
            # pymoo passes its RNG as random_state in **kwargs
            random_state = kwargs.get("random_state", None)
            if random_state is None:
                rng = np.random
            else:
                rng = random_state

            xl = np.asarray(problem.xl, dtype=float)
            xu = np.asarray(problem.xu, dtype=float)

            n_new = n_samples - s
            # uniform in [xl, xu]
            X[s:, :] = rng.random((n_new, n_var)) * (xu - xl) + xl

        return X
