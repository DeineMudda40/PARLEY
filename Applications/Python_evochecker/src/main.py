from evochecker.runner import run_nsga2
import stormpy

def main():
    res = run_nsga2(
        special_prism_path="models/model_10_umc.prism",
        pctl_path="robot.pctl",
        population_size=100,
        max_evaluations=1000,
        n_workers=16,
        rng_seed=42,
        initial_population_path="data/ROBOT10/Seed_Set.tsv",  # or None
    )

    print("Decision vectors X (internal encoding):")
    print(res.X)

    print("Objectives F (minimization; max objectives are negated):")
    print(res.F)

    if res.G is not None:
        print("Constraint violations G (>=0, 0 means satisfied):")
        print(res.G)

    import matplotlib.pyplot as plt

    plt.scatter(res.F[:,0], res.F[:,1])
    plt.xlabel("Objective 1 (negated P)")
    plt.ylabel("Objective 2 (cost)")
    plt.show()



if __name__ == "__main__":
    main()
