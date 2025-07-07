import copy
from collections import defaultdict
import torch
import random

from fl_market.actors.data_consumer import DataConsumer
from fl_market.actors.super_data_consumer import SuperDataConsumer
from fl_market.scenarios.competition import run_competition

from torch.utils.data import ConcatDataset, Subset
from utils.data import filter_dataset


def run_collaboration(
    hps,
    do_datasets,
    dc_valsets,
    public_dataset,
    testset,
    device,
):
    # First, run d_detect rounds in the competitive setting
    hps_comp = copy.deepcopy(hps)
    hps_comp["communication_rounds"] = hps["communication_rounds_before_merging"]
    dcs, dos = run_competition(
        hps_comp,
        do_datasets,
        dc_valsets,
        public_dataset,
        testset,
        device,
    )

    print()
    print("CREATING ALLIANCE...")
    print()

    kept_dcs = dcs[hps["alliance_size"] :]
    dcs = dcs[: hps["alliance_size"]]

    # Get intersection in classes of interest
    coi_intersection = set(dcs[0].classes_of_interest)
    coi_union = set(dcs[0].classes_of_interest)
    for dc in dcs:
        coi_intersection = coi_intersection & set(dc.classes_of_interest)
        coi_union = coi_union | set(dc.classes_of_interest)
    # Create Alliance DC
    alliance_sub_valsets = [filter_dataset(dc.valset, coi_intersection) for dc in dcs]
    alliance_valset = ConcatDataset(alliance_sub_valsets)
    alliance_true_label_mapping = torch.tensor(list(coi_union)).to(device)
    alliance_model = copy.deepcopy(hps["models"][-1]).to(device)

    alliance_dc = DataConsumer(
        "Alliance",
        hps["score_metric"],
        alliance_model,
        list(coi_intersection),
        alliance_valset,
        testset,
        hps["aggregation"],
        public_dataset,
        hps["batch_size"],
        hps["local_epochs"],
        hps["fed_prox_mu"],
        False,  # Not in competition
        device,
        true_label_mapping=alliance_true_label_mapping,
    )
    # Create expert DCs
    expert_dcs = []
    for dc, sub_valset in zip(dcs, alliance_sub_valsets):
        expert_cois = list(set(dc.classes_of_interest) - coi_intersection)
        # Expert valset = old_dc_valset - alliance_valset
        all_dc_val_indices = set(range(len(dc.valset)))
        remaining_indices = all_dc_val_indices - set(sub_valset.indices)
        expert_valset = Subset(dc.valset, list(remaining_indices))
        expert_model = copy.deepcopy(dc.model).to(device)
        # n_in_final_layer = expert_model.linear.in_features
        # n_out_final_layer = len(expert_cois)
        # expert_model.linear = torch.nn.Linear(n_in_final_layer, n_out_final_layer).to(device)
        expert_dcs.append(
            DataConsumer(
                f"Expert-{dc.id}",
                hps["score_metric"],
                expert_model,
                expert_cois,
                expert_valset,
                testset,
                hps["aggregation"],
                public_dataset,
                hps["batch_size"],
                hps["local_epochs"],
                hps["fed_prox_mu"],
                False,  # Not in competition
                device,
                true_label_mapping=dc.true_label_mapping,
            )
        )

    # Create Super DCs

    super_dcs = []

    for dc, expert_dc in zip(dcs, expert_dcs):
        super_dcs.append(
            SuperDataConsumer(
                dc.id,
                hps["score_metric"],
                [expert_dc, alliance_dc],
                copy.deepcopy(dc.model).to(device),
                dc.classes_of_interest,
                dc.valset,
                testset,
                public_dataset,
                dc.test_performances,
                dc.val_performances,
                hps["ekd_epochs"],
                hps["ekd_batch_size"],
                hps["ekd_lr"],
                hps["ekd_temperature"],
                hps["ekd_hard_loss_weight"],
                device,
            )
        )

    # Map DOs to expert DCs

    expert_dcs.append(alliance_dc)
    """dcs_dos = defaultdict(list)
    for dc in expert_dcs:
        for do in dos:
            if do.train_labels.issubset(dc.classes_of_interest):
                dcs_dos[dc].append(do)"""

    recruiting_dcs = kept_dcs + expert_dcs
    print("RECRUITING: ", recruiting_dcs)
    # Data Owners
    dos_interested_dcs = {}
    for do in dos:
        interested_dcs = []
        for dc in recruiting_dcs:
            # Add DO to list of DC if its data is useful to DC
            if do.train_labels.issubset(dc.classes_of_interest):
                interested_dcs.append(dc)
        dos_interested_dcs[do] = interested_dcs

    # Now run the expert DCs in a non-competitive setting and train the SuperDCs on them
    dc_assigned_dos = None
    for round in range(
        hps["communication_rounds_before_merging"] + 1,
        hps["communication_rounds"] + 1,
    ):
        print(f"FL ROUND {round}")
        print()
        """for dc, dos in dcs_dos.items():
            # Every DC can use all DOs in every round
            recruited_dos = dos
            dc.fl_round((recruited_dos, []))"""
        # DC-DO Matching
        print(f"DC-DO Matching")
        if dc_assigned_dos is None or (round - 1) % hps["matching_frequency"] == 0:
            #if "do_assignment" in hps:
            dc_assigned_dos = equal_match_dos_to_dcs(
                dos_interested_dcs, recruiting_dcs
            )
            #else:
            #    dc_assigned_dos = match_dos_to_dcs(dos_interested_dcs)
            for dc, (unique_dos, shared_dos) in dc_assigned_dos.items():
                do_descriptions = [f"(DO{do.id}" for do in shared_dos]
                print(f"DC {dc.id} --> {do_descriptions}")
            print()

        # FL Round
        for dc, dos in dc_assigned_dos.items():
            dc.fl_round(dos)

        for super_dc in super_dcs:
            super_dc.fl_round()

    return super_dcs + kept_dcs, dos


def split_list(lst, n):
    # Divide the list into `n` parts approximately equally
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def match_dos_to_dcs(dos_interested_dcs):
    matching = defaultdict(lambda: ([], []))
    for do, idcs in dos_interested_dcs.items():
        if len(idcs) == 1:
            # Only one dc is interested
            matching[idcs[0]][0].append(do)
        else:
            chosen_dc = random.choice(idcs)
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
