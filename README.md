# FedCDC: A Collaborative Framework for Data Consumers in Federated Learning Markets

This repository contains the code for the paper titled **FedCDC: A Collaborative Framework for Data Consumers in Federated Learning Markets**.

## Requirements
To replicate the results or run the code, you will need the following:

- **Programming Language:** [Python 3.8+]
- **Libraries/Dependencies:** [numpy, pytorch::torch, pytorch::torchvision]

## Usage
By executing run_experiments.py, tests are run for the scenarios with unrestricted and restricted DO-access, as well as with FedCDC. 

 The following arguments are required to run the tests:
* data_path: Folder where the datasets are located (or should be downloaded to).
* result_path: Folder to which the test results will be saved.
* name: Name of the test run.
* seeds: The seeds for which the tests should be executed, e.g. [12, 13]
* dataset: fmnist / cifar10 / cifar100
* num_classes_in_dataset: 10 / 100 depending on dataset
* public_dataset_size: Number of samples in public dataset
* do_train_set_size: Number of training samples of each Data Owner
* dc_val_set_size: Number of validation samples of each Data Consumer
* partitioning: Number of shared and unique classes and corresponding DOs. Implicitly also fixes the number of DCs. E.g. ((3, 6), [(2, 6), (2, 6), (2, 6)]) for 3 Data Consumer with 3 shared and 2 unique classes and 24 Data Owners.
* alliance_size: Number of DCs joining the alliance. For the partitioning above, could be 2 or 3.
* aggregation: fed_avg / fed_df
* communication_rounds: Total number of FL rounds.
* communication_rounds_before_merging: Number of FL rounds before alliance is created.
* matching_frequency: How often DOs are mapped to DCs.
* local_epochs: Training epochs per FL round.
* batch_size: Batch size for local training.
* ekd_lr: Learning rate for ensemble distillation in FedCDC.
* ekd_batch_size: Batch size for ensemble distillation in FedCDC.
* ekd_epochs: Training / Distillation epochs for ensemble distillation in FedCDC.
* ekd_hard_loss_weight: How much weight is put on hard CE-Loss as opposed to soft KL-loss. Corresponds to 1-alpha from the paper.
* ekd_temperature: Temperature used in the soft loss calculation of ensemble distillation. Is set to 1 in all our experiments.

A valid run command could look like this:

```bash
python run_experiments.py -name="test" -data_path="datasets" -result_path="results" -alliance_size=3 -seeds="[12]" -aggregation=fed_avg -communication_rounds=50 -communication_rounds_before_merging=10 -matching_frequency=5 -local_epochs=5 -batch_size=32 -dataset=cifar10 -num_classes_in_dataset=10 -public_dataset_size=5000 -do_train_set_size=1000 -dc_val_set_size=2000 -partitioning="((2, 6), [(2, 6), (2, 6), (2, 6)])" -ekd_lr=0.001 -ekd_batch_size=32 -ekd_epochs=10 -ekd_temperature=1 -ekd_hard_loss_weight=0.0
```
When determining the settings, make sure to keep in mind the size of the training sets (50000 / 60000). 

It is highly recommended to run the code on a GPU. If no GPU is available, change the line
```python
DEVICE = torch.device("cuda")
```
to

```python
DEVICE = torch.device("cpu")
```

in run_experiments.py
