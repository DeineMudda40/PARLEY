import os
from multiprocessing import Pool, cpu_count
import shutil
from Applications.Python_evochecker.src.evochecker.runner import run_nsga2
from Applications.Python_evochecker.src.evochecker.export import (
    save_front_tsv,
    save_parameters_tsv,
    save_hypervolume_history_tsv,
)
import numpy as np


def run(map_, replications, suffix="", load_from_pre_pop=False):

    if load_from_pre_pop:
        init_pop_path = f"data/ROBOT{map_}/Seed_Set"
    else:
        init_pop_path = f"data/ROBOT{map_}/Seed_Set"

    res = run_nsga2(
        special_prism_path=f"models/model_{map_}_umc.prism",
        pctl_path="properties/robot.pctl",
        population_size=160,
        max_evaluations=160 * 250,
        n_workers=cpu_count(),
        rng_seed=42,
        initial_population_path=init_pop_path,
        ref_point=np.array([0.0, 220.0]),
    )

    target_folder = f"data/ROBOT{map_}_REP0_{suffix}"
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)

    os.makedirs(target_folder)

    save_hypervolume_history_tsv(f"data/ROBOT{map_}_REP0_{suffix}/hv_history", res)

    # Save parameter table (decoded ints/doubles/distribution bins)
    save_parameters_tsv(f"data/ROBOT{map_}_REP0_{suffix}/params_out", res)

    # Save objective front (natural values: prob, cost, etc.)
    save_front_tsv(f"data/ROBOT{map_}_REP0_{suffix}/front_out", res)
