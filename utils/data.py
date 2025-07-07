from collections import defaultdict

import numpy as np
import random

from torchvision import transforms, datasets
from torch.utils.data import Subset


# rules=((overlap_size, n_overlap_dos), [(unique_sizes, n_unique_dos)])
def randomly_assign_classes(num_classes, rules):
    all_classes = set(range(num_classes))

    (n_overlap_classes, n_overlap_dos), n_unique_classes_per_dc = rules

    overlap_class_group = random.sample(all_classes, n_overlap_classes)
    all_classes -= set(overlap_class_group)

    assigned_classes = [(overlap_class_group, n_overlap_dos)]

    for class_group_size, n_dos in n_unique_classes_per_dc:
        class_group = random.sample(all_classes, class_group_size)
        all_classes -= set(class_group)
        assigned_classes.append((class_group, n_dos))


    """(n_overlap_classes, n_overlap_dos), n_unique_classes_per_dc = rules

    assigned_do_classes = []
    elms = set(range(num_classes))
    for i in range(num_classes // 2): 
        random_elements = random.sample(elms, 2)
        elms = elms - set(random_elements)
        assigned_do_classes.append((random_elements, 1))

    assigned_dc_classes = []
    shared_classes = []
    i = 0

    for _ in range(n_overlap_classes // 2):
        shared_classes += assigned_do_classes[i][0]
        i += 1
    assigned_dc_classes.append((shared_classes, 1))

    for class_group_size, _ in n_unique_classes_per_dc:
        class_group = []
        for _ in range(class_group_size // 2):
            class_group += assigned_do_classes[i][0]
            i += 1
        assigned_dc_classes.append((class_group, 1)) """

    return assigned_classes


def load_dataset(dataset_name, root):
    # Load dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=transform,
        )
    elif dataset_name == "cifar100":
        dataset = datasets.CIFAR100(
            root=root,
            train=True,
            download=True,
            transform=transform,
        )
    elif dataset_name == "fmnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            ]
        )
        dataset = datasets.FashionMNIST(
            root=root,
            train=True,
            download=True,
            transform=transform,
        )
    else:
        print("ERROR: INVALID DATASET")
        dataset = None
    return dataset


def _partition_dirichlet(class_indices, num_partitions, alpha):
    # Generate Dirichlet probabilities for this class
    dirichlet_probs = np.random.dirichlet(alpha * np.ones(num_partitions))
    # Shuffle class indices to distribute randomly
    np.random.shuffle(class_indices)
    # Partition class indices based on Dirichlet probabilities
    split_indices = np.split(
        class_indices,
        (np.cumsum(dirichlet_probs)[:-1] * len(class_indices)).astype(int),
    )
    return split_indices


def partition_dataset(
    dataset,
    cp_partitioning,
    num_classes,
    partitioning,
    n_public_samples,
    n_do_train_samples,
    n_dc_val_samples,
    dir_alpha,
):

    # Partitioning
    if cp_partitioning is None:
        # Sort sample ids by label
        sample_ids_by_label = defaultdict(list)
        for i, (_, label) in enumerate(dataset):
            sample_ids_by_label[label].append(i)
        # Shuffle sample ids for each label and create public dataset
        public_dataset_sample_ids = np.array([], dtype=int)
        n_public_samples_per_class = int(n_public_samples / num_classes)
        for label, sampleIds in sample_ids_by_label.items():
            np.random.shuffle(sampleIds)
            public_dataset_sample_ids = np.concatenate(
                (public_dataset_sample_ids, sampleIds[:n_public_samples_per_class])
            )
            sample_ids_by_label[label] = sampleIds[n_public_samples_per_class:]

        # Assign samples to each DO group and create partitions from it

        class_groups  = randomly_assign_classes(num_classes, partitioning)
        do_partitions = []

        for labels, n_dos in class_groups:
            group_partitions = [np.array([], dtype=int) for _ in range(n_dos)]
            n_do_classes = len(labels)
            for l in labels:
                n_reserved_samples = int(n_dos * n_do_train_samples / n_do_classes)
                reserved_sample_ids = sample_ids_by_label[l][:n_reserved_samples]
                sample_ids = np.array_split(reserved_sample_ids, n_dos)
                # sample_ids = _partition_dirichlet(reserved_sample_ids, n_dos, dir_alpha)
                sample_ids_by_label[l] = sample_ids_by_label[l][n_reserved_samples:]
                for i, sids in enumerate(sample_ids):
                    group_partitions[i] = np.concatenate((group_partitions[i], sids))
            do_partitions = do_partitions + group_partitions

        for i, p in enumerate(do_partitions):
            if len(do_partitions[i]) % 2 != 0:
                # Make sure that dataset is of even length to prevent that the last batch is of size 1
                # Which would lead to an error
                do_partitions[i] = do_partitions[i][:-1]

        dc_labels = []

        for labels, _ in class_groups[1:]:
            dc_labels.append(labels + class_groups[0][0])

        #print(dc_labels)

        # Load DC val sets
        dc_partitions = [(np.array([], dtype=int), labels) for labels in dc_labels]

        for i, labels in enumerate(dc_labels):
            n_dc_classes = len(labels)
            for l in labels:
                n_reserved_samples = int(n_dc_val_samples / n_dc_classes)
                sample_ids = sample_ids_by_label[l][:n_reserved_samples]
                sample_ids_by_label[l] = sample_ids_by_label[l][n_reserved_samples:]
                dc_partitions[i] = (
                    np.concatenate((dc_partitions[i][0], sample_ids)),
                    dc_partitions[i][1],
                )
    else:
        do_partitions, dc_partitions, public_dataset_sample_ids = cp_partitioning

    #print(dc_partitions)


    # Apply partitioning to dataset
    partitioned_train_set = []
    for ids in do_partitions:
        p_set = Subset(dataset, ids)
        partitioned_train_set.append(p_set)

    partitioned_val_sets = []
    for ids, label_set in dc_partitions:
        p_set = Subset(dataset, ids)
        partitioned_val_sets.append((p_set, label_set))

    public_dataset = Subset(dataset, public_dataset_sample_ids)

    return (
        partitioned_train_set,
        partitioned_val_sets,
        public_dataset,
    )


def filter_dataset(dataset, labels_of_interest):
    indices = [
        idx for idx, (_, label) in enumerate(dataset) if label in labels_of_interest
    ]
    return Subset(dataset, indices)


def load_testset(dataset_name, root):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    if dataset_name == "cifar10":
        testset = datasets.CIFAR10(
            root=root,
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset_name == "cifar100":
        testset = datasets.CIFAR100(
            root=root,
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset_name == "fmnist":
        # Transform the data to match ResNet's expected input size
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            ]
        )
        testset = datasets.FashionMNIST(
            root=root,
            train=False,
            download=True,
            transform=transform,
        )

    return testset
