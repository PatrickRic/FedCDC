import argparse
import ast
import logging
import sys

import random
from datetime import datetime

import numpy as np

import torch


from utils.data import load_testset, partition_dataset, load_dataset
from models.resnet18 import ResNet18
from aggregation.fed_avg import fed_avg
from aggregation.fed_df import fed_df
from fl_market.scenarios.no_competition import run_no_competition
from fl_market.scenarios.competition import run_competition
from fl_market.scenarios.collaboration import run_collaboration


def get_aggregation_technique(name):
    mapping = {"fed_avg": fed_avg}
    return mapping[name]


class PrintLogger:
    """Redirect `print` to logging."""

    def write(self, message):
        if message.strip():  # Avoid empty lines
            logging.info(message)

    def flush(self):
        pass  # Required for compatibility with `sys.stdout`


def setup_logging(runid):
    rp = hps["result_path"]
    logging.basicConfig(
        filename=f"{rp}/{runid}.log",  # Log file
        level=logging.INFO,  # Set log level
        format="%(message)s",  # Format of log messages
    )


def enable_logging():
    sys.stdout = PrintLogger()
    sys.stderr = PrintLogger()  # Redirect errors too


def print_results(dcs, run_name):
    print(f"Results: {run_name}")
    for dc in dcs:
        print()
        print(f"{run_name}_DC_{dc.id}")
        print("VAL: ", dc.val_performances)
        print("TEST: ", dc.test_performances)
        # print("DETAILED: ")
        # print(dc.eval_detailed())
        print()
    print()


def parse_hyperparameters():
    parser = argparse.ArgumentParser(
        description="Parse command line arguments for hyperparameters."
    )

    parser.add_argument(
        "-data_path",
        type=str,
        help="Path where datasets are stored.",
        required=True,
    )

    parser.add_argument(
        "-result_path",
        type=str,
        help="Path where results are stored.",
        required=True,
    )

    parser.add_argument("-seeds", type=str, help="List of seeds", required=True)
    parser.add_argument(
        "-name",
        type=str,
        help="Name of run. Needs to equal name of runai job!",
        required=True,
    )
    parser.add_argument(
        "-score_metric",
        type=str,
        choices=["loss", "accuracy", "contrloss", "greedy_acc"],
        help="Evaluate DC models based on loss or accuracy",
        default="contrloss",
        required=False,
    )
    parser.add_argument(
        "-aggregation", type=str, help="Aggregation method", required=True
    )
    parser.add_argument(
        "-alliance_size",
        type=int,
        help="How many DCs are in alliance?",
        required=True,
    )
    parser.add_argument(
        "-communication_rounds",
        type=int,
        help="Number of communication rounds",
        required=True,
    )
    parser.add_argument(
        "-communication_rounds_before_merging",
        type=int,
        help="Communication rounds before merging",
        required=True,
    )
    parser.add_argument(
        "-local_epochs", type=int, help="Number of local epochs", required=True
    )
    parser.add_argument("-batch_size", type=int, help="Batch size", required=True)
    parser.add_argument("-dataset", type=str, help="Dataset name", required=True)
    parser.add_argument(
        "-num_classes_in_dataset",
        type=int,
        help="Number of classes in dataset. (e.g. 10 for cifar10)",
        required=True,
    )
    parser.add_argument(
        "-public_dataset_size", type=int, help="Public dataset size", required=True
    )
    parser.add_argument(
        "-do_train_set_size",
        type=int,
        help="Training set size for distributed optimization",
        required=True,
    )
    parser.add_argument(
        "-dc_val_set_size",
        type=int,
        help="Validation set size for distributed optimization",
        required=True,
    )
    parser.add_argument(
        "-partitioning", type=str, help="Partitioning strategy", required=True
    )
    """parser.add_argument(
        "-dir_alpha",
        type=float,
        help="Alpha parameter for in-group dirichlet partitioning",
        required=True,
    )"""
    parser.add_argument("-ekd_lr", type=float, help="EKD learning rate", required=True)
    parser.add_argument(
        "-ekd_batch_size", type=int, help="EKD Batch size", required=True
    )
    parser.add_argument("-ekd_epochs", type=int, help="EKD epochs", required=True)
    parser.add_argument(
        "-ekd_temperature", type=float, help="EKD temperature", required=True
    )
    parser.add_argument(
        "-ekd_hard_loss_weight", type=float, help="EKD hard loss weight", required=True
    )
    parser.add_argument(
        "-matching_frequency",
        type=int,
        help="Frequency at which competitive DCs get assigned DOs",
        required=True,
    )

    args = parser.parse_args()

    # Convert parsed arguments to a dictionary
    hyperparameters = vars(args)

    # Post-process the string representations of complex types
    hyperparameters["seeds"] = ast.literal_eval(hyperparameters["seeds"])
    hyperparameters["partitioning"] = ast.literal_eval(hyperparameters["partitioning"])

    return hyperparameters


def print_hyperparameters():
    formatted_hyperparameters = "\n".join(
        [f"{key}: {value}" for key, value in hps.items()]
    )
    print(formatted_hyperparameters)


# RUN

DEVICE = torch.device("cuda")  # "cuda" to train on GPU / "cpu" for CPU
print(f"Training on {DEVICE}")
print(f"PyTorch {torch.__version__}")

# Get and adapt hyperparameters
hps = parse_hyperparameters()
if hps["aggregation"] == "fed_avg":
    hps["aggregation"] = fed_avg
elif hps["aggregation"] == "fed_df":
    hps["aggregation"] = fed_df
else:
    print("INVALID ARGUMENT: aggregation")
hps["fed_prox_mu"] = 0.0

dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
RUNID = dt
cp_partitioning = None
scenario = "no_competition"
init_round = 0

setup_logging(RUNID)

# Run test iteration on all seeds
# For each test iteration: Partition data -> Run: No Competition, Competition, Collaboration
# Store the test performances after every FL round in every scenario

print("Loading data...")
dataset = load_dataset(hps["dataset"], hps["data_path"])
testset = load_testset(hps["dataset"], hps["data_path"])

enable_logging()

print_hyperparameters()

for seed in hps["seeds"]:
    # Apply seed
    print()
    print(f"RUN FOR SEED={seed}")
    print()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set initial models with new seed
    # Initial models for DCs and alliance DC
    ((n_shared_classes, _), dc_unique) = hps["partitioning"]
    n_all_classes = n_shared_classes
    models = []
    for n_unique_classes, _ in dc_unique:
        models.append(
            ResNet18(hps["dataset"], n_shared_classes + n_unique_classes, DEVICE)
        )
        n_all_classes += n_unique_classes
    models.append(ResNet18(hps["dataset"], n_all_classes, DEVICE))
    hps["models"] = models
    # Load training and validation data
    print("Partitioning data")
    do_datasets, dc_valsets, public_dataset = partition_dataset(
        dataset,
        cp_partitioning,
        hps["num_classes_in_dataset"],
        hps["partitioning"],
        hps["public_dataset_size"],
        hps["do_train_set_size"],
        hps["dc_val_set_size"],
        # hps["dir_alpha"],
        0,
    )
    print()

    # Run with no competition
    print("No_Competition")
    print()
    dcs_no_comp, _ = run_no_competition(
        hps,
        do_datasets,
        dc_valsets,
        public_dataset,
        testset,
        DEVICE,
    )
    print_results(dcs_no_comp, "No_Competition")
    # Run with competition
    print("Competition")
    print()
    dcs_comp, _ = run_competition(
        hps,
        do_datasets,
        dc_valsets,
        public_dataset,
        testset,
        DEVICE,
    )
    print_results(dcs_comp, "Competition")
    # Run with collaboration
    print("Collaboration")
    print()
    dcs_collab, _ = run_collaboration(
        hps,
        do_datasets,
        dc_valsets,
        public_dataset,
        testset,
        DEVICE,
    )
    print_results(dcs_collab, "Collaboration")
