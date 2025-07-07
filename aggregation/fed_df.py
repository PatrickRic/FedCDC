from aggregation.ensemble_distillation.ekd import divergence

from aggregation.fed_avg import fed_avg
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader


# Results: List((model, n_samples))
def fed_df(results, trainset, batchsize, device):
    student = fed_avg(results, trainset, batchsize, device)
    teachers = [m for m, _ in results]
    train_loader = DataLoader(trainset, shuffle=True, batch_size=batchsize)
    _fed_df_distillation(
        teachers,
        student,
        train_loader,
        5,  # distillation epochs=5 like in the experiments of the FedHKT paper
        0.001,
        device,
    )
    return student


def _fed_df_distillation(
    teachers,
    student,
    train_loader,
    epochs,
    learning_rate,
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
            avg_teacher_logits = torch.mean(
                torch.stack(teacher_logits).to(device), dim=0
            )
            loss = divergence(student_logits, avg_teacher_logits, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
