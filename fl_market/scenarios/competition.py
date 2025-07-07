from collections import defaultdict
import random
import copy

from fl_market.actors.data_consumer import DataConsumer
from fl_market.actors.data_owner import DataOwner


def run_competition(
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
    dcs = []
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
            True,  # Is in competition
            device,
        )
        dcs.append(dc)

    print()
    # Data Owners
    dos_interested_dcs = {}
    for id, trainset in enumerate(do_datasets):
        do = DataOwner(id, trainset, device)
        interested_dcs = []
        for dc in dcs:
            # Add DO to list of DC if its data is useful to DC
            if do.train_labels.issubset(dc.classes_of_interest):
                interested_dcs.append(dc)
        dos_interested_dcs[do] = interested_dcs

    print()

    # RUN SIMULATION
    dc_assigned_dos = None
    for round in range(1, hps["communication_rounds"] + 1):
        print(f"FL ROUND {round}")
        print()
        # DC-DO Matching
        print(f"DC-DO Matching")
        if dc_assigned_dos is None or (round - 1) % hps["matching_frequency"] == 0:
            #if "do_assignment" in hps:
            dc_assigned_dos = equal_match_dos_to_dcs(dos_interested_dcs, dcs)
            #else:
            #    dc_assigned_dos = match_dos_to_dcs(dos_interested_dcs)
            for dc, (unique_dos, shared_dos) in dc_assigned_dos.items():
                do_descriptions = [f"(DO{do.id}" for do in shared_dos]
                print(f"DC {dc.id} --> {do_descriptions}")
            print()

        # FL Round
        for dc, dos in dc_assigned_dos.items():
            dc.fl_round(dos)

    return dcs, dos_interested_dcs.keys()


def split_list(lst, n):
    # Divide the list into `n` parts approximately equally
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def match_dos_to_dcs(dos_interested_dcs):
    matching = defaultdict(lambda: ([], []))
    for do, dcs in dos_interested_dcs.items():
        if len(dcs) == 1:
            # Only one dc is interested
            matching[dcs[0]][0].append(do)
        else:
            chosen_dc = random.choice(dcs)
            matching[chosen_dc][1].append(do)

    return matching


# dos_interested_dcs: Dictionary mapping every data owner to all interested data consumers
# Returns: Dictionary of form: DC -> (unique_dos, shared_dos)
def equal_match_dos_to_dcs(dos_interested_dcs, dcs):
    shared = []
    matching = defaultdict(lambda: ([], []))
    for do, do_dcs in dos_interested_dcs.items():
        if len(do_dcs) == 1:
            # Only one dc is interested
            matching[do_dcs[0]][0].append(do)
        else:
            shared.append(do)
    random.shuffle(shared)
    for sdos, dc in zip(split_list(shared, len(dcs)), dcs):
        matching[dc] = (matching[dc][0], sdos)

    return matching
