import torch
import numpy as np
import copy


def train_model(net, global_net, trainloader, true_label_mapping, epochs, mu, device):
    # Train the network on the training set.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)
            converted_labels = torch.tensor(
                [(true_label_mapping == l).nonzero(as_tuple=True)[0] for l in labels]
            ).to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), converted_labels)
            prox_loss = 0.0
            if not mu is None:
                for param, global_param in zip(net.parameters(), global_net.parameters()):
                    prox_loss += (mu / 2) * torch.sum((param - global_param) ** 2)

            loss += prox_loss

            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += converted_labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == converted_labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        # epoch_acc = correct / total
        # if DETAILED_LOG and (epoch == 0 or epoch == epochs - 1):
        #    print(
        #        f"{description}  Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}"
        #   )


def test_model(net, testloader, true_label_mapping, device):
    # Evaluate the network on the entire test set.
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)
            converted_labels = torch.tensor(
                [(true_label_mapping == l).nonzero(as_tuple=True)[0] for l in labels]
            ).to(device)
            outputs = net(images)
            loss += criterion(outputs, converted_labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += converted_labels.size(0)
            correct += (predicted == converted_labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return accuracy, loss


 # Returns true labels and predictions as numpy arrays
def get_predictions(model, data_loader, true_label_mapping, device):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            preds = true_label_mapping[preds]
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)
