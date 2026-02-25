import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

def pareto_front(data):
    pareto_data = []
    for x, y in data:
        if not is_dominated(x, y, pareto_data, data):
            pareto_data.append((x, y))
    return pareto_data


def is_dominated(x, y, data1, data2):
    for other_x, other_y in data1:
        # Check if another point is better in both objectives: maximize x and minimize y
        if other_x >= x and other_y <= y:
            # Strict domination: at least one condition must be strictly better
            if other_x > x or other_y < y:
                return True
    for other_x, other_y in data2:
        # Check if another point is better in both objectives: maximize x and minimize y
        if other_x >= x and other_y <= y:
            # Strict domination: at least one condition must be strictly better
            if other_x > x or other_y < y:
                return True
    return False


def plot_pareto_front(m=10, replication=0, header=True):
    file_path = (
        f"Applications/EvoChecker-master/data/ROBOT{m}_REP{replication}_PLUS/NSGAII/"
    )
    x_values_pp, y_values_pp = __get_data(file_path, header)

    file_path = f"Applications/EvoChecker-master/data/ROBOT{m}_REP{replication}/NSGAII/"
    x_values_p, y_values_p = __get_data(file_path, header)

    filepath = f"Applications/EvoChecker-master/data/ROBOT{m}_BASELINE"
    x_values_b, y_values_b = __get_data(filepath, header=False, split="	")
    x_values_b = x_values_b[:10]
    y_values_b = y_values_b[:10]

    plt.figure(figsize=(8, 6))

    # Add blue dots for Parley
    plt.scatter(
        x_values_p,
        y_values_p,
        facecolors="none",
        edgecolors="green",
        marker="o",
        label="PARLEY",
    )
    # Add red crosses for the baseline
    plt.scatter(x_values_b, y_values_b, color="red", marker="x", label="Baseline")
    # Add green pluses for Parley
    plt.scatter(x_values_pp, y_values_pp, color="blue", marker="+", label="PARLEY+")

    plt.xlabel("Probability of mission success")
    plt.ylabel("Cost")
    output_filename = f"robot{m}_rep{replication}"
    plt.title(output_filename)
    plt.xlim(1, 0.2)
    plt.ylim(0, 200)
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file
    plt.savefig("plots/fronts/" + output_filename + ".pdf")
    plt.close()


def plot_pareto_front_aware(m=10, replication=0, header=True, targets_fronts=[]):
    plt.figure(figsize=(8, 6))

    colors = ["blue", "green", "orange"]
    markers = ["D", "^", "p"]  # circle, square, triangle
    point_size = 20  # smaller than default (default ≈ 36)

    for i, front in enumerate(targets_fronts):
        file_path = f"data/ROBOT{m}_REP{replication}_{front}/front_out"
        x_values, y_values = __get_data(file_path, header)

        plt.scatter(
            x_values,
            y_values,
            label=front,
            color=colors[i],
            marker=markers[i],
            s=point_size,
        )

    filepath = f"data/ROBOT{m}/Seed_Front"
    x_values_b, y_values_b = __get_data(
        filepath, header=False, split="\t", filter_pareto_front=False
    )

    x_values_b = x_values_b[:10]
    y_values_b = y_values_b[:10]

    plt.scatter(
        x_values_b,
        y_values_b,
        color="red",
        marker="x",
        s=40,  # make baseline a bit more visible
        label="Baseline",
    )

    plt.xlabel("Probability of mission success")
    plt.ylabel("Cost")
    output_filename = f"robot{m}_rep{replication}"
    # plt.title(output_filename)

    plt.xlim(0.4, 0.2)
    plt.legend()
    plt.grid(True)

    plt.savefig("plots/fronts/" + output_filename + ".png")
    plt.close()


def plot_pareto_front_for_robot(m=10, replication=2, ref_point=(0.7, 80), header=True):
    plt.rcParams.update({"font.size": 15})

    file_path = (
        f"Applications/EvoChecker-master/data/ROBOT{m}_REP{replication}_PLUS/NSGAII/"
    )
    x_values_pp, y_values_pp = __get_data(file_path, header)

    file_path = f"Applications/EvoChecker-master/data/ROBOT{m}_REP{replication}/NSGAII/"
    x_values_p, y_values_p = __get_data(file_path, header)

    filepath = f"Applications/EvoChecker-master/data/ROBOT{m}_BASELINE"
    x_values_b, y_values_b = __get_data(filepath, header=False, split="	")
    x_values_b = x_values_b[:10]
    y_values_b = y_values_b[:10]

    plt.figure(figsize=(10, 6))

    # Add red crosses for the baseline
    plt.scatter(x_values_b, y_values_b, color="red", marker="x", label="Baseline")
    # Add blue dots for Parley
    plt.scatter(
        x_values_p,
        y_values_p,
        facecolors="none",
        edgecolors="green",
        marker="o",
        label="PARLEY",
    )
    # Add green pluses for Parley
    plt.scatter(x_values_pp, y_values_pp, color="blue", marker="+", label="PARLEY+")

    # Plot dashed lines at the ref_point (x=probability, y=cost)
    plt.axvline(
        ref_point[0],
        color="gray",
        linestyle="--",
        label="Minimum Allowed Success Probability",
    )
    plt.axhline(
        ref_point[1], color="gray", linestyle="--", label="Maximum Allowed Cost"
    )

    # Shading only below ref_point on the x-axis (left side)
    plt.fill_betweenx(
        [0, ref_point[1]], x1=0.2, x2=ref_point[0], color="gray", alpha=0.2
    )  # Left side

    # Shading only above ref_point on the y-axis (top side)
    plt.fill_between(
        [1, 0.2], y1=ref_point[1], y2=200, color="gray", alpha=0.2
    )  # Top side

    # Highlight maximum x-values whose y-value is below the ref point
    def highlight_max_below_ref(x_values, y_values, color, label):
        # Filter data points below the reference point in y
        below_ref_points = [
            (x, y) for x, y in zip(x_values, y_values) if y <= ref_point[1]
        ]
        if below_ref_points:
            # Find the maximum x-value among them
            max_x_point = max(below_ref_points, key=lambda point: point[0])
            # Plot the max point with a special marker and label
            plt.scatter(
                max_x_point[0],
                max_x_point[1],
                color=color,
                s=100,
                marker="D",
                label=f"Chosen policy with {label}",
            )  # Custom label here

    # Highlight the best points for each approach
    highlight_max_below_ref(x_values_b, y_values_b, "red", "Baseline")
    highlight_max_below_ref(x_values_p, y_values_p, "green", "PARLEY")
    highlight_max_below_ref(x_values_pp, y_values_pp, "blue", "PARLEY+")

    plt.xlabel("Probability of mission success")
    plt.ylabel("Cost")
    output_filename = f"robot{m}_rep{replication}"
    plt.xlim(1, 0.5)
    plt.ylim(0, 120)
    plt.legend()  # Moves the legend to a custom location

    plt.grid(True)
    # plt.show()
    # Save the plot as an image file
    plt.savefig("front.pdf", bbox_inches="tight")
    plt.close()


def plot_pareto_front_for_gazebo(ref_point=(0.2, 30), header=True):
    plt.rcParams.update({"font.size": 16})
    file_path = f"Applications/EvoChecker-master/data/gazebo/PARLEY+/"
    x_values_pp, y_values_pp = __get_data(file_path, header)

    file_path = f"Applications/EvoChecker-master/data/gazebo/PARLEY/"
    x_values_p, y_values_p = __get_data(file_path, header)

    file_path = f"Applications/EvoChecker-master/data/gazebo/baseline/"
    x_values_b, y_values_b = __get_data(file_path, header)
    x_values_b = x_values_b[:5]
    y_values_b = y_values_b[:5]

    plt.figure(figsize=(8, 6))

    # Add red crosses for the baseline
    plt.scatter(x_values_b, y_values_b, color="red", marker="x", label="Baseline")
    # Add blue dots for Parley
    plt.scatter(
        x_values_p,
        y_values_p,
        facecolors="none",
        edgecolors="green",
        marker="o",
        label="PARLEY",
    )
    # Add green pluses for Parley
    plt.scatter(x_values_pp, y_values_pp, color="blue", marker="+", label="PARLEY+")

    # Plot dashed lines at the ref_point (x=probability, y=cost)
    plt.axvline(ref_point[0], color="gray", linestyle="--", label="Minimum Probability")
    plt.axhline(
        ref_point[1], color="gray", linestyle="--", label="Maximum Available Cost"
    )

    # Shading only below ref_point on the x-axis (left side)
    plt.fill_betweenx(
        [0, ref_point[1]], x1=0, x2=ref_point[0], color="gray", alpha=0.2
    )  # Left side

    # Shading only above ref_point on the y-axis (top side)
    plt.fill_between(
        [1, 0], y1=ref_point[1], y2=200, color="gray", alpha=0.2
    )  # Top side

    # Highlight maximum x-values whose y-value is below the ref point
    def highlight_max_below_ref(x_values, y_values, color, label):
        # Filter data points below the reference point in y
        below_ref_points = [
            (x, y) for x, y in zip(x_values, y_values) if y <= ref_point[1]
        ]
        if below_ref_points:
            # Find the maximum x-value among them
            max_x_point = max(below_ref_points, key=lambda point: point[0])
            # Plot the max point with a special marker and label
            plt.scatter(
                max_x_point[0],
                max_x_point[1],
                color=color,
                s=100,
                marker="D",
                label=f"Chosen policy with {label}",
            )  # Custom label here

    # Highlight the best points for each approach
    highlight_max_below_ref(x_values_b, y_values_b, "red", "Baseline")
    highlight_max_below_ref(x_values_p, y_values_p, "green", "PARLEY")
    highlight_max_below_ref(x_values_pp, y_values_pp, "blue", "PARLEY+")

    plt.xlabel("Probability of mission success")
    plt.ylabel("Cost")
    output_filename = "gazebo"
    plt.xlim(0.3, 0)
    plt.ylim(0, 50)
    plt.legend()  # Moves the legend to a custom location

    plt.grid(True)
    # plt.show()
    # Save the plot as an image file
    plt.savefig("front_gazebo.pdf", bbox_inches="tight")
    plt.close()


def plot_pareto_front_for_tas(ref_point=(10, 75), header=True):
    plt.rcParams.update({"font.size": 16})

    file_path = f"Applications/EvoChecker-master/data/TAS/PARLEY+/"
    x_values_pp, y_values_pp = __get_data(file_path, header)

    file_path = f"Applications/EvoChecker-master/data/TAS/PARLEY/"
    x_values_p, y_values_p = __get_data(file_path, header)

    file_path = f"Applications/EvoChecker-master/data/TAS/baseline/"
    x_values_b, y_values_b = __get_data(file_path, header)
    x_values_b = x_values_b[:10]
    y_values_b = y_values_b[:10]

    plt.figure(figsize=(8, 6))

    # Add red crosses for the baseline
    plt.scatter(x_values_b, y_values_b, color="red", marker="x", label="Baseline")
    # Add blue dots for Parley
    plt.scatter(
        x_values_p,
        y_values_p,
        facecolors="none",
        edgecolors="green",
        marker="o",
        label="PARLEY",
    )
    # Add green pluses for Parley
    plt.scatter(x_values_pp, y_values_pp, color="blue", marker="+", label="PARLEY+")

    # Plot dashed lines at the ref_point (x=probability, y=cost)
    plt.axvline(
        ref_point[0], color="gray", linestyle="--", label="Minimum Correct Alarms"
    )
    plt.axhline(
        ref_point[1], color="gray", linestyle="--", label="Maximum Available Cost"
    )

    # Shading only below ref_point on the x-axis (left side)
    plt.fill_betweenx(
        [0, ref_point[1]], x1=0, x2=ref_point[0], color="gray", alpha=0.2
    )  # Left side

    # Shading only above ref_point on the y-axis (top side)
    plt.fill_between(
        [20, 0], y1=ref_point[1], y2=200, color="gray", alpha=0.2
    )  # Top side

    # Highlight maximum x-values whose y-value is below the ref point
    def highlight_max_below_ref(x_values, y_values, color, label):
        # Filter data points below the reference point in y
        below_ref_points = [
            (x, y) for x, y in zip(x_values, y_values) if y <= ref_point[1]
        ]
        if below_ref_points:
            # Find the maximum x-value among them
            max_x_point = max(below_ref_points, key=lambda point: point[0])
            # Plot the max point with a special marker and label
            plt.scatter(
                max_x_point[0],
                max_x_point[1],
                color=color,
                s=100,
                marker="D",
                label=f"Chosen policy with {label}",
            )  # Custom label here

    # Highlight the best points for each approach
    highlight_max_below_ref(x_values_b, y_values_b, "red", "Baseline")
    highlight_max_below_ref(x_values_p, y_values_p, "green", "PARLEY")
    highlight_max_below_ref(x_values_pp, y_values_pp, "blue", "PARLEY+")

    plt.xlabel("Correct Alarms")
    plt.ylabel("Cost")
    output_filename = "TAS"
    plt.xlim(15, 0)
    plt.ylim(0, 180)
    plt.legend()  # Moves the legend to a custom location

    plt.grid(True)
    # plt.show()
    # Save the plot as an image file
    plt.savefig("front_tas.pdf", bbox_inches="tight")
    plt.close()


def __get_hv_history(file_path, header=True, split="\t"):
    evals = []
    hv_values = []

    with open(file_path, "r") as file:
        if header:
            next(file)  # skip header
        for line in file:
            parts = line.strip().split(split)

            # Expect at least n_eval and hv
            n_eval = float(parts[0])
            hv = float(parts[1])

            evals.append(n_eval)
            hv_values.append(hv)

    return np.array(evals), np.array(hv_values)


def __get_data(file_path, header, split="\t", filter_pareto_front=True):
    data = []
    with open(file_path, "r") as file:
        if header:
            next(file)  # Skip the header row
        for line in file:
            x, y = map(float, line.strip().split(split))
            data.append((x, y))
    if filter_pareto_front:
        pareto_data = pareto_front(data)
    else:
        pareto_data = data
    x_values = [x for x, y in pareto_data]
    y_values = [y for x, y in pareto_data]
    return x_values, y_values


# plot_pareto_front()
# plot_pareto_front_for_robot()
# plot_pareto_front_for_gazebo()
# plot_pareto_front_for_tas()


def plot_hypervolume_over_evals(
    m=10,
    replication=0,
    targets_fronts=[],
    n_individuals=160,
):
    """
    Plot hypervolume progression over evaluations for multiple approaches.
    """

    plt.figure(figsize=(8, 6))

    colors = ["blue", "green", "orange", "purple"]
    markers = ["o", "^", "s", "D"]

    for i, front in enumerate(targets_fronts):
        file_path = f"data/ROBOT{m}_REP{replication}_{front}/hv_history"

        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping.")
            continue

        evals, hv_values = __get_hv_history(file_path)

        # Optional: enforce monotonicity for nicer plots
        # hv_values = np.maximum.accumulate(hv_values)

        plt.plot(
            evals/n_individuals,
            hv_values,
            label=front,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            markersize=4,
            linewidth=1.5,
        )

    plt.xlabel("Number of Generations")
    plt.ylabel("Hypervolume")
    output_filename = f"robot{m}_rep{replication}_hv"
    # plt.title(output_filename)

    #ax=plt.gca()
    #ax.ticklabel_format(style="sci", axis="x",scilimits=(0,0))

    plt.legend()
    plt.grid(True)

    plt.savefig("plots/hypervolume/" + output_filename + ".png")
    plt.close()
