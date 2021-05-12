import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import time
import json
import pandas

from opacus import PrivacyEngine
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image

# Own modules
import v6simplemodel as sm
import util.parser as parser

from vantage6.tools.util import info, warn


def master(client, data, *args, **kwargs): #central algorithm uses the methods of node_algorithm
    """Master algorithm.
    The master algorithm is the chair of the Round Robin, which makes
    sure everyone waits for their turn to identify themselves.
    """
    # Info messages can help you when an algorithm crashes. These info
    # messages are stored in a log file which is send to the server when
    # either a task finished or crashes.
    info('Collecting participating organizations')

    # Collect all organization that participate in this collaboration.
    # These organizations will receive the task to compute the partial.
    organizations = client.get_organizations_in_my_collaboration()
    ids = [organization.get("id") for organization in organizations]

    # Request all participating parties to compute their partial. This
    # will create a new task at the central server for them to pick up.
    # We've used a kwarg but is is also possible to use `args`. Although
    # we prefer kwargs as it is clearer.
    info('Requesting partial computation')
    task = client.create_new_task(
        input_={
            'method': 'average_partial',
            'kwargs': {
                'initialize' : initialize_training() # or what goes here?
            }
        },
        organization_ids=ids
    )

    # Now we need to wait until all organizations(/nodes) finished
    # their partial. We do this by polling the server for results. It is
    # also possible to subscribe to a websocket channel to get status
    # updates.
    info("Waiting for results")
    task_id = task.get("id")
    task = client.get_task(task_id)
    while not task.get("complete"):
        task = client.get_task(task_id)
        info("Waiting for results")
        time.sleep(1)

    # Once we now the partials are complete, we can collect them.
    info("Obtaining results")
    results = client.get_results(task_id=task.get("id"))

    model = sm.Net()

    # TODO federated averaging:
        # TODO Calculate the average of the parameters and adjust global model
        # client.create_new_task sends to nodes
        # results = client.get_results(task_id=task.get("id")) gets results from nodes

    parser.parse_arguments()


# # TODO send average parameters weighted to workers like client.send

def average_parameters_weighted(model, parameters, weights):
    with torch.no_grad():
        for param in model.parameters():
            average = sum(x * y for x, y in zip(parameters[i], weights)) / sum(weights)
            param.data = average
            i = i + 1
        return parameters

# def fed_avg(rank, color, model, device, args, group, optimizer, train_loader,
#     test_loader, weights):
#     """
#     Training and testing the model on the workers concurrently using federated
#     averaging, which means calculating the average of the local model
#     parameters after a number of (local) epochs each training round.
#
#     Args:
#         rank: The id of the process.
#         color: The color for the terminal output for this worker.
#         model: A model to run training on.
#         device: The device to run training on.
#         args: The parsed arguments.
#         group: The world group.
#         optimizer: Optimization algorithm used for training.
#         train_loader: Data loader for training data.
#         test_loader: The data loader for test data.
#         weights: A list of weights for each worker, representing the size of the
#         dataset at the workernode, will only be used by the server.
#
#     Returns:
#         Returns the final model
#     """


## TODO: gather paramerters which gathers all the new model parameters from the workers and broadcast after

# TODO DATA !! -> send to nodes full dataset or sample and do indexing at node

###### NODE SECTION: These functions will be run at node and coordinated through master function; Prefix "RPC_" is required

def RPC_initialize_training(rank, group, color, args):
    """
    Initializes the model, optimizer and scheduler and shares the parameters
    with all the workers in the group.

    Args:
        rank: The id of the process.
        group: The group the process belongs to.
        color: The color for the terminal output for this worker.
        learning_rate: The learning rate for training.
        cuda: Should we use CUDA?

    Returns:
        Returns the device, model, optimizer and scheduler.
    """
    # Determine the device to train on
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("\033[0;{};49m Rank {} is training on {}".format(color, rank, device))

    # Initialize model and send parameters of server to all workers
    model = sm.Net().to(device)
    # coor.broadcast_parameters(model, group) # this will be done in master function

    # Intializing optimizer and scheduler

    # comment S: use opacus for DP: Opacus is a library that enables training PyTorch models with differential privacy. Taken from: https://github.com/pytorch/opacus
    # makes dp.py obsolete for this project

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    if args.local_dp:
        privacy_engine = PrivacyEngine(model, batch_size=64,
            sample_size=60000, alphas=range(2,32), noise_multiplier=1.3,
            max_grad_norm=1.0,)
        privacy_engine.attach(optimizer)

    return device, model, optimizer



def RPC_train(rank, color, log_interval, model, device, train_loader, optimizer,
    epoch, round, local_dp, delta=1e-5):
    """
    Training the model on all batches.
    Args:
        rank: The id of the process.
        color: The color for the terminal output for this worker.
        log_interval: The amount of rounds before logging intermediate loss.
        model: A model to run training on.
        device: The device to run training on.
        train_loader: Data loader for training data.
        optimizer: Optimization algorithm used for training.
        epoch: The number of the epoch the training is in.
        round: The number of the round the training is in.
        local_dp: Training with local DP?
        delta: The delta value of DP to aim for (default: 1e-5).
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Calculate the loss
        batch = (data, target)
        loss = train_batch(model, device, batch, optimizer)
        # Log information once every log interval
        if batch_idx % log_interval == 0:
            print('\033[0;{};49m Train on Rank {}, Round {}, Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                color, rank, round, epoch, batch_idx * len(batch[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    if local_dp:
        epsilon, alpha = optimizer.privacy_engine.get_privacy_spent(delta)
        print("\033[0;{};49m Epsilon {}, best alpha {}".format(color, epsilon, alpha))


def RPC_test(rank, color, model, device, test_loader):
    """
    Tests the model.

    Args:
        rank: The id of the process.
        color: The color for the terminal output for this worker.
        model: The model to test.
        device: The device to test the model on.
        test_loader: The data loader for test data.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Send the data and target to the device (cpu/gpu) the model is at
            data, target = data.to(device), target.to(device)
            # Run the model on the data
            output = model(data)
            # Calculate the loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # Check whether prediction was correct
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\033[0;{};49m \nTest set on Rank {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        color, rank, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def RPC_train_batch(model, device, batch, optimizer, train=True):
    """
    Training the model on one batch of data.

    Args:
        model: A model to run training on.
        device: The device to run training on.
        batch: The batch to train the model on.
        optimizer: Optimization algorithm used for training.
        train: Should we update the model parameters? (default:true)

    Returns:
        The calculated loss after training.
    """
    data, target = batch
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

    # Update the model weights
    if train:
        optimizer.step()
    return loss