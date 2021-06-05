"""
Author: Simon Tokloth
Date:
Description: This module contains the RPC_methods including the training and federated averaging.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import transforms

# Own modules
from .v6simplemodel import Net

# ----NODE-----
# RPC_methods always need to start with local
# since parameters of fed_Avg are node-dependent, it should happen in a RPC call; also: everything that is dependeant on local should happen in RPC_call
# if don't want to use local in RPC call: RPC_init_training(_, rank, ...) maybe


def RPC_initialize_training(data, color, args):
    """
    Initializes the model, optimizer and scheduler and shares the parameters
    with all the workers in the group.

    This should be sent from server to all nodes.

    Args:
        color: The color for the terminal output for this worker.
        learning_rate: The learning rate for training.
        cuda: Should we use CUDA?

    Returns:
        Returns the device, model, optimizer and scheduler.
    """
    # Load local dataset
    # first:
        # torch.save(dataset_train, './dataset.pt')
    data = torch.load('./dataset.pt')

    # Determine the device to train on
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # print("\033[0;{};49m Rank {} is training on {}".format(color, rank, device))

    # Initialize model and send parameters of server to all workers
    model = Net()
    model.to(device)

    # TODO: load local? train_loader, test_loader from locally stored local

    # use Opacus for DP: Opacus is a library that enables training PyTorch models
    # with differential privacy. Taken from: https://github.com/pytorch/opacus

    # Intializing optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    if args.local_dp:
        privacy_engine = PrivacyEngine(model, batch_size=64,
            sample_size=60000, alphas=range(2,32), noise_multiplier=1.3,
            max_grad_norm=1.0)
        privacy_engine.attach(optimizer)

    return device, model, optimizer


## TODO: QUESTION: If RPC_train is only used by node method (Fed_avg), does it need the local parameter? Or can it just be def train(color, model, ...)?

def RPC_train(data, color, model, device, train_loader, optimizer, epoch,
    local_dp, delta=1e-5):
    """
    Training the model on all batches.
    Args:
        color: The color for the terminal output for this worker.
        model: A model to run training on.
        device: The device to run training on.
        train_loader: Data loader for training local.
        optimizer: Optimization algorithm used for training.
        epoch: The number of the epoch the training is in.
        round: The number of the round the training is in.
        local_dp: Training with local DP?
        delta: The delta value of DP to aim for (default: 1e-5).
    """
    # TODO: define train_loader again from local local
    data = torch.load('./dataset.pt')
    train_loader, test_loader, data_size = data

    model.train()

    for i, (data, target) in enumerate(train_loader):
        # Send the local and target to the device (cpu/gpu) the model is at (either send to the cpu or to the gpu, but the local is already on the worker node); model.send(local.location)
        data, target = data.to(device), target.to(device)
        # Clear gradient buffers
        optimizer.zero_grad()
        # Run the model on the local
        output = model(data)
        # Calculate the loss
        loss = F.nll_loss(output, target)
        # Calculate the gradients
        loss.backward()
        optimizer.step()
    return epoch
    # Logging needed if want the same output as torch.dist
    # print('\033[0;{};49m Train on Rank {}, Round {}, Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #     color, rank, round, epoch, batch_idx * len(batch[0]), len(train_loader.dataset),
    #     100. * batch_idx / len(train_loader), loss.item()))

    if local_dp:
        epsilon, alpha = optimizer.privacy_engine.get_privacy_spent(delta)
        print("\033[0;{};49m Epsilon {}, best alpha {}".format(color, epsilon, alpha))


def RPC_test(data, color, model, device, test_loader):
    """
    Tests the model.

    Args:
        color: The color for the terminal output for this worker.
        model: The model to test.
        device: The device to test the model on.
        test_loader: The local loader for test local.
    """
    # TODO: load local dataset as test_loader
    data = torch.load('./dataset.pt')
    train_loader, test_loader, data_size = data

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

    print('\033[0;{};49m \nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        color, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


#-----FED_AVG------

# TODO federated averaging:

# def RPC_get_parameters(data, model, parameters, weights):
#     """
#     Get parameters from nodes
#     """
#
# "for parameters in nodes:
#       return parameters"


def RPC_average_parameters_weighted(data, model, parameters, weights):
    """
    Get parameters from nodes and calculate the average
    :param model: torch model
    :param parameters: parameters of model
    :param weights:
    :return:
    """
    # TODO: local: since we usually just get the parameters, this well be an entire task, therefore, we might need to train for each individually

    with torch.no_grad():
        for parameters in model.parameters():
            average = sum(x * y for x, y in zip(parameters[i], weights)) / sum(weights)
            parameters.data = average
            i = i + 1
        return parameters

def RPC_fed_avg(data, color, args, model, optimizer, train_loader, test_loader, device):
    """
    Training and testing the model on the workers concurrently using federated
    averaging, which means calculating the average of the local model
    parameters after a number of (local) epochs each training round.


    Returns:
        Returns the final model
    """
    # TODO: local: since we usually just get the parameters, this well be an entire task, therefore, we might need to train for each individually

    for epoch in range(1, args.epochs + 1):
        # Train the model on the workers
        RPC_train(data, color, model, device, train_loader, optimizer, epoch, args.local_dp, delta=1e-5)
        # Test the model on the workers
        RPC_test(data, color, model, device, test_loader)

    gather_params = model.get_parameters()

    model.RPC_average_parameters_weighted(gather_params)

    return model

    # # Gather the parameters after the training round on the server
    #     gather_params = coor.gather_parameters(rank, model, group_size + 1, subgroup)
    #
    #     # If the server
    #     if rank == 0:
    #         # Calculate the average of the parameters and adjust global model
    #         coor.average_parameters_weighted(model, gather_params, weights)
    #
    #     # Send the new model parameters to the workers
    #     coor.broadcast_parameters(model, group)


