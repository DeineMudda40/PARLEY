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


def models(i, uncertainty_aware=False, param_file="input.json"):
    prism_model_generator.generate_robot_model(i, param_file=param_file)
    infile = f"Applications/EvoChecker-master/models/model_{i}.prism"
    outfile = f"Applications/EvoChecker-master/models/model_{i}_umc.prism"
    os.makedirs(f"Applications/EvoChecker-master/data/ROBOT{i}", exist_ok=True)
    popfile = f"Applications/EvoChecker-master/data/ROBOT{i}/Front"

    with open(param_file, "r") as json_file:

        data = json.load(json_file)
        min_val = data["min_val"]
        max_val = data["max_val"]

    if uncertainty_aware:
        urc_synthesis.ParleyUAMealy(
            infile,
            internal_states=5,
            transition_after_update=False,
            min_val=min_val,
            max_val=max_val,range_split=True
        ).transform_file(infile, outfile, popfile)
    else:
        urc_synthesis.ParleyPlusURC(
            infile, transition_after_update=False, min_val=min_val, max_val=max_val
        ).transform_file(infile, outfile, popfile)


def baseline(i):
    baseline_file = f"Applications/EvoChecker-master/data/ROBOT{i}_BASELINE/Front"
    # baseline_file = f'Applications/EvoChecker-master/data/TAS/baseline/Front'
    infile = f"Applications/EvoChecker-master/models/model_{i}.prism"
    # infile = f'Applications/EvoChecker-master/models/TAS/TAS.prism'
    os.makedirs(f"Applications/EvoChecker-master/data/ROBOT{i}_BASELINE", exist_ok=True)
    with open(baseline_file, "w") as b_file:
        for period in range(1, 11):
            b_file.write(prism_caller.compute_baseline(infile, period))
            if period < 10:
                b_file.write("\n")
            print("finished baseline map {0}, value {1}".format(str(i), str(period)))


def evo_checker(i, suffix=""):
    # invoke EvoChecker
    run_evochecker.run(i, max_replications, suffix)


def fronts(i):
    for period in range(max_replications):
        # plot_fronts.plot_pareto_front(i, period)
        plot_fronts.plot_pareto_front_aware(i, period)


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
    __modify_properties()
    # maps()
    models(i, uncertainty_aware=False)
    baseline(i)
    evo_checker(i, "_PLUS")


def run_aware(i):
    #__modify_properties()

    models(i, uncertainty_aware=True)
    # baseline(i)
    evo_checker(i, "_UA")


def run_countdown(i):
    models(i, uncertainty_aware=True, param_file="input.json")
    #baseline(i)
    evo_checker(i, "_PLUS")


def run_binary(i):
    models(i, uncertainty_aware=False, param_file="input_binary.json")
    # baseline(i)
    evo_checker(i, "_UA")

def run_(i):
    models(i, uncertainty_aware=False, param_file="input_binary.json")
    # baseline(i)
    evo_checker(i, "_UA")


def main2():
    # maps()
    for i in range(10, 11):

        run_countdown(i)
        #run_binary(i)

        #run_unaware(i)
        #run_aware(i)
        fronts(i)


if __name__ == "__main__":
    os.makedirs("plots/fronts", exist_ok=True)
    os.makedirs("plots/box-plots", exist_ok=True)
    main2()
