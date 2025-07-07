import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

import numpy as np

from utils.data import filter_dataset

import statistics


# HELPER FUNCTIONS


def divergence(student_logits, teacher_logits, temperature, dim=1, reduction="mean"):
    divergence = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=dim),
        F.softmax(teacher_logits / temperature, dim=dim),
        reduction=reduction,
    )  # forward KL

    if reduction == "none":
        divergence = divergence.mean(dim=dim)

    return divergence


def entropy(logits, dim=2):
    probs = F.softmax(logits, dim=dim)
    return -torch.sum(
        probs * torch.log(probs + 1e-10), dim=dim
    )  # Small epsilon to prevent log0 error


def avg_logits(logits):
    # Calculate weights
    entropies = entropy(logits)
    alphas_unnormalized = torch.exp(-entropies)
    sum_alphas = alphas_unnormalized.sum(dim=0, keepdim=True)
    alphas_normalized = alphas_unnormalized / sum_alphas
    weights = alphas_normalized.unsqueeze(2)
    # Compute weighted average
    return torch.sum(weights * logits, dim=0)


def loss_student(
    student_logits, avg_teacher_logits, T, hard_loss_weight, reduction="mean"
):
    soft_loss = divergence(student_logits, avg_teacher_logits, T, reduction=reduction)
    pseudo_labels = torch.argmax(avg_teacher_logits, dim=1)
    hard_loss = F.cross_entropy(student_logits, pseudo_labels, reduction=reduction)
    epsilon = hard_loss_weight
    return epsilon * hard_loss + (1 - epsilon) * soft_loss


def get_teacher_to_student_mappings(teacher_dcs, student_mapping):
    teacher_to_student_mappings = [
        [-1 for _ in tdc.true_label_mapping] for tdc in teacher_dcs
    ]
    for student_index, label in enumerate(student_mapping):
        for i, teacher in enumerate(teacher_dcs):
            teacher_mapping = teacher.true_label_mapping.tolist()
            try:
                teacher_index = teacher_mapping.index(label)
                teacher_to_student_mappings[i][teacher_index] = student_index
            except ValueError:
                # DC does not use the label
                pass
    return teacher_to_student_mappings


def adjust_teacher_logits(
    student_labels_shape,
    teacher_to_student_output_mappings,
    teacher_logits,
    device,
):
    adjusted_teacher_logits = []
    for mapping, logits in zip(teacher_to_student_output_mappings, teacher_logits):
        v_not_present = -1000.0
        adj_logits = torch.full(student_labels_shape, v_not_present).to(device)
        batch_size = student_labels_shape[0]
        for batch in range(batch_size):
            for i in range(len(mapping)):
                # Only map, if there is an equivalent in the student output
                if mapping[i] >= 0:
                    adj_logits[batch, mapping[i]] = logits[batch][i]
        adjusted_teacher_logits.append(adj_logits)

    # Only works for two teachers, for which the union of the labels they predict is a superset of the labels that the student predicts
    adjusted_teacher_logits[0] = torch.where(
        adjusted_teacher_logits[0] == v_not_present,
        adjusted_teacher_logits[1],
        adjusted_teacher_logits[0],
    )
    adjusted_teacher_logits[1] = torch.where(
        adjusted_teacher_logits[1] == v_not_present,
        adjusted_teacher_logits[0],
        adjusted_teacher_logits[1],
    )

    adjusted_teacher_logits = torch.stack(adjusted_teacher_logits).to(device)

    return adjusted_teacher_logits


# EKD


def perform_ensemble_distillation(
    teachers,
    student,
    teacher_to_student_output_mappings,
    train_loader,
    epochs,
    learning_rate,
    T,
    hard_loss_weight,
    device,
    verbose=False,
):
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    [teacher.eval() for teacher in teachers]  # Teachers set to evaluation mode
    student.train()  # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)

            optimizer.zero_grad()

            # Forward pass with the student model
            student_logits = student(inputs)
            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = [teacher(inputs) for teacher in teachers]
            # Convert ouput logits so that they can be used for student training
            adjusted_teacher_logits = adjust_teacher_logits(
                student_logits.shape,
                teacher_to_student_output_mappings,
                teacher_logits,
                device,
            )
            # print(teacher_logits[1][0], "->", adjusted_teacher_logits[1][0])
            avg_teacher_logits = avg_logits(adjusted_teacher_logits)

            loss = loss_student(student_logits, avg_teacher_logits, T, hard_loss_weight)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")


def perform_ensemble_distillation_with_filtering(
    teachers,
    student,
    teacher_to_student_output_mappings,
    train_set,
    batch_size,
    epochs,
    learning_rate,
    T,
    hard_loss_weight,
    selection_frequency,
    selection_ratio,
    device,
    verbose=False,
):
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    [teacher.eval() for teacher in teachers]  # Teachers set to evaluation mode
    student.train()  # Student to train model

    for epoch in range(epochs):
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        running_loss = 0.0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)

            optimizer.zero_grad()

            # Forward pass with the student model
            student_logits = student(inputs)
            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = [teacher(inputs) for teacher in teachers]
            # Convert ouput logits so that they can be used for student training
            adjusted_teacher_logits = adjust_teacher_logits(
                student_logits.shape,
                teacher_to_student_output_mappings,
                teacher_logits,
                device,
            )
            avg_teacher_logits = avg_logits(adjusted_teacher_logits)

            loss = loss_student(student_logits, avg_teacher_logits, T, hard_loss_weight)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
        if epoch % selection_frequency == 5:
            # Filter public dataset
            filter_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
            all_losses = torch.zeros((len(train_set), 3), device=device)
            for batch_idx, (inputs, _) in enumerate(filter_loader):
                inputs = inputs.to(device)
                # Calculate loss
                with torch.no_grad():
                    teacher_logits = [teacher(inputs) for teacher in teachers]
                    student_logits = student(inputs)
                # Convert ouput logits so that they can be used for student training
                adjusted_teacher_logits = adjust_teacher_logits(
                    student_logits.shape,
                    teacher_to_student_output_mappings,
                    teacher_logits,
                    device,
                )
                avg_teacher_logits = avg_logits(adjusted_teacher_logits)
                losses = loss_student(
                    student_logits,
                    avg_teacher_logits,
                    T,
                    hard_loss_weight,
                    reduction="none",
                )
                pseudo_labels = torch.argmax(avg_teacher_logits, dim=1)
                start_idx = batch_idx * batch_size
                end_idx = start_idx + len(losses)
                in_batch_idx = torch.tensor(range(len(losses))).to(device)
                global_idx = start_idx + in_batch_idx
                all_losses[start_idx:end_idx, 0] = pseudo_labels
                all_losses[start_idx:end_idx, 1] = losses
                all_losses[start_idx:end_idx, 2] = global_idx
            # Filtr each classes
            all_losses = all_losses.cpu()
            filtered_losses = []
            for c in np.unique(all_losses[:, 0]):  # Loop over unique types
                # Filter rows for the current type
                c_rows = all_losses[all_losses[:, 0] == c]
                # Sort by score (second column)
                sorted_rows = c_rows[np.argsort(c_rows[:, 1])]
                sorted_ids = sorted_rows[:, 2]
                # Keep only the top fraction of rows
                num_to_keep = int(np.ceil(selection_ratio * len(sorted_ids)))
                filtered_losses.append(sorted_ids[:num_to_keep])

            # Combine the results into a single array
            filtered_losses = np.concatenate(filtered_losses)
            train_set = Subset(train_set, filtered_losses.astype(int))
