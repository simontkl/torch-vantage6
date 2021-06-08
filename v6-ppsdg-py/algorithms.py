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




## TODO: QUESTION: If RPC_train is only used by node method (Fed_avg), does it need the local parameter? Or can it just be def train(color, model, ...)?

# basic training of the model

# Question: train gets model, device, optimizer from initialize_training, which is specified within train function,
# why do I need to call it again before executing the function? Because in vantage6 when I sent the tasks I cannot define that but only in the master function


def RPC_train_test(data, data2, device, model, optimizer, log_interval, local_dp, epoch, delta=1e-5):
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

    device, model, optimizer = initialize_training(gamma, learning_rate, local_dp)

    train_loader = data

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

#-----FED_AVG------

# TODO federated averaging:

# FedAvg gathering of parameters

def RPC_get_parameters(data, model):
    """
    Get parameters from nodes
    """

    with torch.no_grad():
        torch.manual_seed(1)
        for parameters in model.parameters():
            # store parameters in dict
            return {"params": parameters}

"""
this might need to be combined with training, so that train 
returns the parameters or that it at least calls the results of training function
"""


# training with those averaged parameters

def RPC_fed_avg(data, model, local_dp, epoch, delta=1e-5):
    """
    Training and testing the model on the workers concurrently using federated
    averaging, which means calculating the average of the local model
    parameters after a number of (local) epochs each training round.

    In vantage6, this method will be the training of the model with the average parameters (weighted)

    Returns:
        Returns the final model
    """

    # train and test with new parameters
    for round in range(1, round + 1):
        for epoch in range(1, epoch + 1):
            # Train the model on the workers again
            RPC_train_test()


