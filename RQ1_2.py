import json
import os

import create_maps
import evaluation
import prism_model_generator
import prism_caller
import run_evochecker
import plot_fronts
import urc_synthesis2
import urc_synthesis

max_replications = 1


def maps():
    create_maps.create_90_maps()


def models(i, uncertainty_aware=False, param_file="input.json",mode="normal"):
    prism_model_generator.generate_robot_model(i, param_file=param_file)
    infile = f"models/model_{i}.prism"
    outfile = f"models/model_{i}_umc.prism"
    os.makedirs(f"data/ROBOT{i}", exist_ok=True)
    popfile = f"data/ROBOT{i}/Seed_Set"

    with open(param_file, "r") as json_file:

        data = json.load(json_file)
        min_val = data["min_val"]
        max_val = data["max_val"]

    if uncertainty_aware:
        urc_synthesis2.ParleyFSCMealyCircleDist(
            infile, internal_states=5, min_val=min_val, max_val=max_val
        ).transform_file(infile, outfile,popfile)
    else:
        if mode=="normal":
            urc_synthesis.ParleyPlusURC(
                infile, transition_after_update=False, min_val=min_val, max_val=max_val
            ).transform_file(infile, outfile, popfile)

def baseline(i):
    baseline_file = f"data/ROBOT{i}/Seed_Front"
    infile = f"models/model_{i}.prism"
    with open(baseline_file, "w") as b_file:
        for period in range(1, 11):
            output=prism_caller.compute_baseline(infile, period)
            b_file.write(output)
            if period < 10:
                b_file.write("\n")
            print(f"finished baseline map {i}, value {period}, {output}")


def evo_checker(i, uncertainty_aware, suffix=""):
    # invoke EvoChecker
    run_evochecker.run(i, max_replications, suffix)


def fronts(i,targets_fronts=["PLUS","DIST"]):
    for period in range(max_replications):
        # plot_fronts.plot_pareto_front(i, period)
        plot_fronts.plot_pareto_front_aware(i, period,targets_fronts=targets_fronts)
        plot_fronts.plot_hypervolume_over_evals(i,period,targets_fronts=targets_fronts)


def run_unaware(i):
    models(i, uncertainty_aware=False,mode="normal")
    baseline(i)
    evo_checker(i, False, "PLUS")

def run_unaware_dist(i):
    models(i, uncertainty_aware=False,mode="distribution")
    evo_checker(i, False, "DIST")


def run_aware(i):
    models(i, uncertainty_aware=True)
    evo_checker(i, False, "UA")


def run_aware(i):
    models(i, uncertainty_aware=True)
    #evo_checker(i, False, "EXTENDED")

def main2():
    #maps()
    import time
    for i in range(10, 11):
        
        #run_unaware(i)
        #run_aware(i)
        fronts(i,targets_fronts=["PLUS","EXTENDED"])
        #fronts(i,targets_fronts=[])


if __name__ == "__main__":
    os.makedirs("plots/fronts", exist_ok=True)
    os.makedirs("plots/box-plots", exist_ok=True)
    main2()
