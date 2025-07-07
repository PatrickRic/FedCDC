import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.ticker import FormatStrFormatter
import re
import sys
from matplotlib import rcParams
from cycler import cycler
from matplotlib.ticker import MaxNLocator


import numpy as np

num_rounds = 50
num_rounds += 1

if len(sys.argv) < 2:
    print("Error! No run specified in command line arguments!")
    sys.exit()

type = sys.argv[1]

use_loss = 1 if "-loss" in sys.argv else 0  # 1 if loss should be used
metric = "Loss" if use_loss == 1 else "Accuracy"

types = ["FedAvg", "FedDF"]
t = ["fedavg", "feddf"]
log_path = [f"final/{type}-{run}.log" for run in t]


def get_results(file_path, n_rounds):
    # Open and read the file content
    with open(file_path, "r") as file:
        file_content = file.read()
        file_content = file_content.replace(
            "\xa0", " "
        )  # Replace non-breaking spaces with regular spaces

    # Regular expression to match test performances
    def parse_results(pattern):
        performance_pattern = pattern
        # performance_pattern = r"\[([^\]]+)\]"

        # Extracting all the performance arrays
        performance_matches = re.findall(performance_pattern, file_content, re.DOTALL)

        # Parse each performance array into lists of tuples (accuracy, loss)
        performances = []
        for match in performance_matches:
            # Convert string representation of list to actual list of tuples
            values = eval(
                f"[{match}]"
            )  # Use eval carefully, only if content format is trusted
            performances.append(values[0])

        return performances

    # Extracting performances
    test_performances = parse_results(r"TEST:\s*\n(\[[^\]]+\])")
    # for i, t in enumerate(test_performances):
    # print(len(t), " ", i, i // (3 * 3))
    test_performances = np.array(test_performances)
    # Didn't log first 10 val performances for collaboration DCs...
    val_performances = parse_results(r"VAL:\s*\n(\[[^\]]+\])")
    print("VAL:")
    for i, t in enumerate(val_performances):
        print(len(t), " ", i, i // (3 * 3))
    print("---")

    """for i in range(len(val_performances)):
        if i % 9 == 6 or i % 9 == 7 or i % 9 == 8:
            val_performances[i] = np.concatenate(
                (test_performances[i, :10, :], np.array(val_performances[i]))
            )"""

    # print(len(val_performances[i]))

    val_performances = np.array(val_performances)
    # For each seed: 3 Runs with 3 DCs
    n_results_per_seed = 3 * 3

    n_seeds = int(len(test_performances) / n_results_per_seed)
    print(n_seeds)

    for seed_index in range(n_seeds):
        base_index = n_results_per_seed * seed_index
        for run_index in range(3):
            average_run_results = np.zeros(n_rounds)
            for dc_index in range(3):
                test_results = test_performances[
                    base_index + 3 * run_index + dc_index, :, use_loss
                ]
                val_results = val_performances[
                    base_index + 3 * run_index + dc_index, :, use_loss
                ]
                curr_best = 100.0 if use_loss == 1 else 0.0
                curr_v = 100.0 if use_loss == 1 else 0.0
                for i, (v, r) in enumerate(zip(val_results, test_results)):
                    if (use_loss == 1 and v < curr_v) or (use_loss == 0 and v > curr_v):
                        curr_v = v
                        curr_best = r
                    else:
                        test_performances[
                            base_index + 3 * run_index + dc_index, i, use_loss
                        ] = curr_best

    average_results = np.zeros((3, n_rounds))
    max_results = np.zeros((3, n_seeds))  # For one seed (otherwise for last seed)
    for seed_index in range(n_seeds):
        base_index = n_results_per_seed * seed_index
        for run_index in range(3):
            average_run_results = np.zeros(n_rounds)
            for dc_index in range(3):
                dc_run_results = (
                    test_performances[
                        base_index + 3 * run_index + dc_index,
                        :,
                        use_loss,
                    ],
                )[
                    0
                ]  # Need this because of autoformatting :/
                average_run_results += dc_run_results
            average_run_results /= 3
            average_results[run_index] += average_run_results
            max_results[run_index, seed_index] = average_run_results[-1]
    average_results /= n_seeds

    return average_results, max_results


results = []
maxres = []

for i in range(len(t)):
    print(log_path[i], " ...")
    r, mr = get_results(log_path[i], num_rounds)
    # print("X: ", mr)
    results.append(r)
    maxres.append(mr)

# Plotting
dark_colors = ["#00008B", "#B22222", "#228B22"]
plt.rc("axes", prop_cycle=cycler("color", dark_colors))
# Customize the appearance
rcParams["font.family"] = "Times New Roman"
rcParams["font.family"] = "Times New Roman"
rcParams["xtick.labelsize"] = 22  # Adjust size as needed
rcParams["ytick.labelsize"] = 22

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(27, 6))
fig.subplots_adjust(left=0.07, right=0.9, top=0.9, bottom=0.15, wspace=0.1)


for i, aggr in enumerate(["FedAvg", "FedDF"]):
    for j, scen in enumerate(["Unrestricted", "Restricted", "Collab"]):
        max_values_per_seed = maxres[i][j]
        mean = 100 * np.mean(max_values_per_seed)
        var = 100 * np.var(max_values_per_seed)
        print(f"{aggr} {scen} : ${mean:.2f} \\pm {var:.2f}$")


for i, r in enumerate(types):

    n_rounds = num_rounds
    rounds = list(range(n_rounds))

    fmt = lambda x, pos: str(int(x))
    axes[i].xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))

    # Plot the data
    axes[i].plot(
        rounds,
        results[i][0],
        label="Unrestricted DO-Access",
        linewidth=5,
        linestyle=":",
        # marker="^",
        markersize=6,
    )
    axes[i].plot(
        rounds,
        results[i][1],
        label="Restricted DO-Access",
        linewidth=5,
        linestyle="--",
        # marker="s",
        markersize=6,
    )
    # axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[i].plot(
        rounds,
        results[i][2],
        label="FedCDC",
        linewidth=5,
        # marker="o",
        markersize=6,
    )

    # Title and labels
    axes[i].set_title(
        types[i], fontsize=22, fontweight="bold", fontfamily="Times New Roman"
    )
    # Add grid
    # axes[i].grid(alpha=0.4, linestyle="--")  # Use dashed gridlines for subtlety
    # Legend
    axes[i].legend(fontsize=20, loc="lower right", frameon=True, edgecolor="black")

fig.supxlabel("Rounds", fontsize=22)
fig.supylabel(metric, fontsize=22)

plt.subplots_adjust()
# plt.tight_layout()
plt.savefig(f"temp.pdf", bbox_inches="tight")
plt.show()
