import copy
import random

from fl_market.actors.data_consumer import DataConsumer
from fl_market.actors.data_owner import DataOwner


def run_no_competition(
    hps,
    do_datasets,
    dc_valsets,
    public_dataset,
    testset,
    device,
):
    # CREATE DATA OWNERS AND DATA CONSUMERS

    # Data Consumers

    #  Dictionary of the form DataConsumer -> List Of DataOwners used by this DataConsumer
    dcs = {}
    for i, (valset, cois) in enumerate(dc_valsets):
        dc = DataConsumer(
            i,
            hps["score_metric"],
            copy.deepcopy(hps["models"][i]).to(device),
            cois,
            valset,
            testset,
            hps["aggregation"],
            public_dataset,
            hps["batch_size"],
            hps["local_epochs"],
            hps["fed_prox_mu"],
            False,  # Not in competition
            device,
        )
        dcs[dc] = []

    print()
    # Data Owners
    dos = []
    for id, trainset in enumerate(do_datasets):
        do = DataOwner(id, trainset, device)
        dos.append(do)
        for dc in dcs.keys():
            # Add DO to list of DC if its data is useful to DC
            if do.train_labels.issubset(dc.classes_of_interest):
                dcs[dc].append(do)
    print()

    # RUN SIMULATION
    for round in range(1, hps["communication_rounds"] + 1):
        print(f"FL ROUND {round}")
        print()
        for dc, dos_for_dc in dcs.items():
            # Every DC can use all DOs in every round
            dc.fl_round((dos_for_dc, []))

    return dcs.keys(), dos
