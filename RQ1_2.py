import json
import os

import create_maps
import evaluation
import prism_model_generator
import prism_caller
import run_evochecker
import plot_fronts
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
        urc_synthesis.ParleyFSCMealy(
            infile, internal_states=2, min_val=min_val, max_val=max_val
        ).transform_file(infile, outfile)
    else:
        if mode=="normal":
            urc_synthesis.ParleyPlusURC(
                infile, transition_after_update=False, min_val=min_val, max_val=max_val
            ).transform_file(infile, outfile, popfile)
        elif mode=="distribution":
            urc_synthesis.ParleyPlusURCDist(
                infile, transition_after_update=False, min_val=min_val, max_val=max_val
            ).transform_file(infile, outfile, popfile)

def baseline(i):
    baseline_file = f"data/ROBOT{i}/Seed_Front"
    infile = f"models/model_{i}.prism"
    with open(baseline_file, "w") as b_file:
        for period in range(1, 11):
            b_file.write(prism_caller.compute_baseline(infile, period))
            if period < 10:
                b_file.write("\n")
            print("finished baseline map {0}, value {1}".format(str(i), str(period)))


def evo_checker(i, uncertainty_aware, suffix=""):
    # invoke EvoChecker
    run_evochecker.run(i, max_replications, suffix)


def fronts(i,targets_fronts=["PLUS","DIST"]):
    for period in range(max_replications):
        # plot_fronts.plot_pareto_front(i, period)
        plot_fronts.plot_pareto_front_aware(i, period,targets_fronts=targets_fronts)


def __modify_properties():
    try:
        with open("input.json", "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {
            "startX": 0,
            "startY": 0,
            "targetX": 9,
            "targetY": 9,
            "p": 0.01,
            "updates": [5],
            "map_file": "map.csv",
        }
    # Modify targetX and targetY to be equal to i
    data["targetX"] = 9
    data["targetY"] = 9
    # Write the modified data back to the JSON file
    with open("input.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


def run_unaware(i):
    models(i, uncertainty_aware=False,mode="normal")
    baseline(i)
    evo_checker(i, False, "_PLUS")

def run_unaware_dist(i):
    models(i, uncertainty_aware=False,mode="distribution")
    evo_checker(i, False, "_DIST")


def run_aware(i):
    models(i, uncertainty_aware=True)
    evo_checker(i, False, "_UA")


def main2():
    #maps()
    for i in range(10, 11):

        run_unaware(i)
        run_unaware_dist(i)
        fronts(i,targets_fronts=["PLUS","DIST"])


if __name__ == "__main__":
    os.makedirs("plots/fronts", exist_ok=True)
    os.makedirs("plots/box-plots", exist_ok=True)
    main2()
