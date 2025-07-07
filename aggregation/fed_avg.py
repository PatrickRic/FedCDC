import copy

import torch


# Aggregates results using FedAvg
# Also averages non-trainable parameters (e.g. batch-normalization statistics)
# It may be beneficial to use a silo approach for non-trainable parameters
# Results: List((model, n_samples))
def fed_avg(results, public_dataset, batch_size, device):
    if len(results) == 0:
        return None

    # Return as Pytorch model
    aggregated_model = copy.deepcopy(results[0][0]).to(device)
    aggregated_state_dict = aggregated_model.state_dict()

    # Reset all parameter values in the aggregated model to zero
    int64_layers = []
    for key in aggregated_state_dict:
        if aggregated_state_dict[key].dtype == torch.int64:
            # We also want to average int parameters (such as batch-normalization statistics), so we turn them into floats
            # We store the keys of the int-layers, so that we can turn the average back into an int lateron
            int64_layers.append(key)
            aggregated_state_dict[key] = torch.zeros_like(
                aggregated_state_dict[key], dtype=torch.float64
            ).to(device)
        else:
            aggregated_state_dict[key] = torch.zeros_like(
                aggregated_state_dict[key]
            ).to(device)

    # Calculate the total number of samples across all models
    total_samples = sum(n_samples for _, n_samples in results)

    # Accumulate the weighted parameters
    for model, n_samples in results:
        model_state_dict = model.state_dict()
        for key in model_state_dict:
            aggregated_state_dict[key] += model_state_dict[key] * (
                n_samples / total_samples
            )

    # Convert converted int64 layers back to int64
    for key in int64_layers:
        aggregated_state_dict[key] = aggregated_state_dict[key].round().to(torch.int64)
    # Load the weighted average parameters into the aggregated model
    aggregated_model.load_state_dict(aggregated_state_dict)

    return aggregated_model
