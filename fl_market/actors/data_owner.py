import copy
from collections import Counter

from torch.utils.data import DataLoader


class DataOwner:
    def __init__(self, id, trainset, device):
        self.id = id
        self.trainset = trainset

        self.train_labels = set([label for (_, label) in trainset])
        """
        label_counts = Counter()
        for _, label in trainset:  # Assuming the dataset returns (data, label)
            label_counts[label.item()] += 1  # Convert tensor to integer and count
        # Convert to a dictionary (optional)
        self.sample_distribution = dict(label_counts)
        """
        self.n_samples = len(trainset)
        self.device = device

        print(f"D0{id}: {self.n_samples} samples from classes {self.train_labels}")

    def fit(self, global_model, true_label_mapping, batch_size, epochs, mu):
        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

        model = copy.deepcopy(global_model).to(self.device)

        model.train_model(global_model, trainloader, true_label_mapping, epochs, mu)
        return model, self.n_samples
