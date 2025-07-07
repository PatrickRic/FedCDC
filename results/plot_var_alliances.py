import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.ticker import FormatStrFormatter
import re
import sys
from matplotlib import rcParams
from cycler import cycler


import numpy as np

num_rounds = 50
num_rounds += 1

n_dc = 4

if len(sys.argv) < 2:
    print("Error! No run specified in command line arguments!")
    sys.exit()

type = sys.argv[1]

use_loss = 1 if "-loss" in sys.argv else 0  # 1 if loss should be used
metric = "Loss" if use_loss == 1 else "Accuracy"

types = ["FedAvg"]  # , "FedProx", "FedDF"]
t = ["c10_var_alliances"]  # , "fedprox", "feddf"]
log_path = [f"{run}.log" for run in t]


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
        print(len(t), " ", i, i // (4 * n_dc))
    print("---")

    # print(len(val_performances[i]))

    val_performances = np.array(val_performances)
    # For each seed: 3 Runs with 3 DCs

    n_seeds = 10
    # n_seeds = 1
    n_dcs_per_scenario = [4, 2, 3, 4]

    n_runs = 4
    n_results_per_run = n_seeds * n_dc
    print("LVP: ", len(val_performances))

    average_results = np.zeros((n_runs, n_rounds))
    max_results = np.zeros((n_runs, n_seeds))  # For one seed (otherwise for last seed)

    for run_index in range(n_runs):
        base_index = n_results_per_run * run_index
        average_run_results = np.zeros(n_rounds)
        for seed_index in range(n_seeds):
            for dc_index in range(n_dc):
                test_results = test_performances[
                    base_index + n_dc * seed_index + dc_index, :, use_loss
                ]
                val_results = val_performances[
                    base_index + n_dc * seed_index + dc_index, :, use_loss
                ]
                curr_best = 100.0 if use_loss == 1 else 0.0
                curr_v = 100.0 if use_loss == 1 else 0.0
                for i, (v, r) in enumerate(zip(val_results, test_results)):
                    if (use_loss == 1 and v < curr_v) or (use_loss == 0 and v > curr_v):
                        curr_v = v
                        curr_best = r
                    else:
                        test_performances[
                            base_index + n_dc * seed_index + dc_index, i, use_loss
                        ] = curr_best

    average_results = np.zeros((n_runs, n_rounds))
    max_results = np.zeros((n_runs, n_seeds))  # For one seed (otherwise for last seed)

    for run_index in range(n_runs):
        base_index = n_results_per_run * run_index
        for seed_index in range(n_seeds):
            average_run_results = np.zeros(n_rounds)
            for dc_index in range(n_dc):
                dc_run_seed_results = (
                    test_performances[
                        base_index + n_dc * seed_index + dc_index,
                        :,
                        use_loss,
                    ],
                )[
                    0
                ]  # Need this because of autoformatting :/
                if dc_index < n_dcs_per_scenario[run_index]:
                    average_run_results += dc_run_seed_results
            average_run_results /= n_dcs_per_scenario[run_index]
            average_results[run_index] += average_run_results
    average_results /= n_seeds

    return average_results, max_results


results = []
maxres = []

for i in range(len(t)):
    print(log_path[i], " ...")
    r, mr = get_results(log_path[i], num_rounds)
    print(r.shape)
    results.append(r)
    maxres.append(mr)

# Plotting
dark_colors = ["#00008B", "#B22222", "#228B22", "#000000"]
plt.rc("axes", prop_cycle=cycler("color", dark_colors))
# Customize the appearance
rcParams["font.family"] = "Times New Roman"
rcParams["font.family"] = "Times New Roman"
rcParams["xtick.labelsize"] = 22  # Adjust size as needed
rcParams["ytick.labelsize"] = 22


for i, aggr in enumerate(["FedAvg"]):
    for j, scen in enumerate(["Unrestricted", "Restricted", "Collab"]):
        max_values_per_seed = maxres[i][j]
        mean = 100 * np.mean(max_values_per_seed)
        var = 100 * np.var(max_values_per_seed)
        print(f"{aggr} {scen} : ${mean:.2f} \\pm {var:.2f}$")

plt.figure(figsize=(20, 6))

final_results = []

print(results[0].shape)

for i in range(4):
    final_results.append(results[0][i][-1])

num_alliance_sizes = 4
alliance_sizes = list(range(num_alliance_sizes))

fmt = lambda x, pos: str(int(x + 1) if x != 0 else "No Alliance")
plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
plt.xticks([0, 1, 2, 3])

plt.plot(
    [0, 1, 2, 3],
    final_results,
    label="No Alliance",
    linewidth=5,
    # linestyle=":",
    marker="o",
    markersize=12,
)


plt.savefig(f"var_alliances.pdf", bbox_inches="tight")
plt.show()
