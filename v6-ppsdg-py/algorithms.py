"""
Author: Simon Tokloth
Date:
Description: This module contains the RPC_methods including the training and federated averaging.
"""

import torch
import torch.nn.functional as F

# Own modules
from .central import initialize_training

# ----NODE-----
# RPC_methods always need to start with local
# since parameters of fed_Avg are node-dependent, it should happen in a RPC call; also: everything that is dependeant on local should happen in RPC_call
# if don't want to use local in RPC call: RPC_init_training(_, rank, ...) maybe


# basic training of the model

def RPC_train_test(data, **kwargs):
    """
    Training the model on all batches.
    Args:
        epoch: The number of the epoch the training is in.
        round: The number of the round the training is in.
        log_interval: The amount of rounds before logging intermediate loss.
        local_dp: Training with local DP?
        delta: The delta value of DP to aim for (default: 1e-5).
        data: dataset for train_loader will need to be specified here
        data2: dataset for test_loader will need to be specified here
    """
    # loading arguments/parameters from first RPC_method

    device, model, optimizer = initialize_training(0.01, False)

    log_interval = 10

    epoch = 1

    local_dp = False

    train_loader = data

    data2 =  torch.load(
        "C:\\Users\\simon\\PycharmProjects\\torch-vantage6\\v6-ppsdg-py\\local\\MNIST\\processed\\testing.pt")

    test_loader = data2

    model.train()

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

    # Adding differential privacy or not
    if local_dp == True:
        epsilon, alpha = optimizer.privacy_engine.get_privacy_spent(delta)
    #             print("\033[0;{};49m Epsilon {}, best alpha {}".format(epsilon, alpha))

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
