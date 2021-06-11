"""
Author: Simon Tokloth
Date:
Description: This module contains the RPC_methods including the training and federated averaging.
"""

import torch
import torch.nn.functional as F
from opacus import PrivacyEngine
import torch.optim as optim



# Own modules


# basic training and testing of the model
# from torch import nn


def RPC_train_test(data, model, parameters, test_loader, optimizer, device, log_interval, local_dp, epoch, round, delta):
    """
    Training the model on all batches.
    Args:
        epoch: The number of the epoch the training is in.
        model: use model from initialize_training, if it doesn't work try all separately and change model params in fed_avg
        parameters:
        test_loader: test dataset which is separate
        optimizer:
        device:
        log_interval: The amount of rounds before logging intermediate loss.
        local_dp: Training with local DP?
        epoch
        round
        delta: The delta value of DP to aim for (default: 1e-5).
    """

    train_loader = data

    new_params = model.parameters(parameters)

    optimizer = optim.SGD(new_params, lr=0.01, momentum=0.5)

    model.train()
    for epoch in range(1, epoch + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Send the data and target to the device (cpu/gpu) the model is at
            data, target = data.to(device), target.to(device)
            # Clear gradient buffers
            optimizer.zero_grad()
            # Run the model on the data
            output = model(data)
            # Calculate the loss
            loss = F.nll_loss(output, target)
            # Calculate the gradients
            loss.backward()
            # Update model
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

        torch.save(model, f"C:\\Users\\simon\\PycharmProjects\\torch-vantage6\\v6-ppsdg-py\\local\\model_trained.pth")

        model.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                # Send the local and target to the device (cpu/gpu) the model is at
                data, target = data.to(device), target.to(device)
                # Run the model on the local
                output = model(data)
                # Calculate the loss
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # Check whether prediction was correct
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset)))

# Gathering of parameters for federated averaging
def RPC_get_parameters(data, model):
    """
    Get parameters from nodes
    """

    # with torch.no_grad():
    for parameters in model.parameters():
        # store parameters in dict
        return {"params": parameters}


"""
Experimentation
"""
# for RPC_get_parameters:

# new_params = OrderedDict()
#
# n = len(clients)  # number of clients
#
# for client_model in clients:
#   sd = client_model.state_dict()  # get current parameters of one client
#   for k, v in sd.items():
#     new_params[k] = new_params.get(k, 0) + v / n

# cannot access client like that. model.parameters(), but maybe:

    # new_params = OrderedDict()
    #
    # n = len(organizations)
    #
    # for model in model.parameters():
    #     sd = model.state_dict()
    #     for k, v in sd.items():
    #         new_params[k] = new_params.get(k, 0) + v / n