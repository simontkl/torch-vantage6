"""
Author: Simon Tokloth
Date:
Description: This module contains the RPC_methods including the training and federated averaging.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine

# Own modules
from .v6simplemodel import Net


# ----NODE-----
# RPC_methods always need to start with local
# since parameters of fed_Avg are node-dependent, it should happen in a RPC call; also: everything that is dependeant on local should happen in RPC_call
# if don't want to use local in RPC call: RPC_init_training(_, rank, ...) maybe


def RPC_initialize_training(data, gamma, learning_rate, local_dp):
    """
    Initializes the model, optimizer and scheduler and shares the parameters
    with all the workers in the group.

    This should be sent from server to all nodes.

    Args:
        data: contains the local data from the node
        gamma: Learning rate step gamma (default: 0.7)
        learning_rate: The learning rate for training.
        cuda: Should we use CUDA?
        local_dp: bool whether to apply local_dp or not.

    Returns:
        Returns the device, model, optimizer and scheduler.
    """

    # Determine the device to train on
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # print("\033[0;{};49m Rank {} is training on {}".format(device))

    # Initialize model and send parameters of server to all workers
    model = Net()
    model.to(device)

    # intializing optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # adding DP if true
    if local_dp == True:
        privacy_engine = PrivacyEngine(model, batch_size=64,
                                       sample_size=60000, alphas=range(2, 32), noise_multiplier=1.3,
                                       max_grad_norm=1.0, )
        privacy_engine.attach(optimizer)

    # returns device, model, optimizer which will be needed in train and test
    return device, model, optimizer


## TODO: QUESTION: If RPC_train is only used by node method (Fed_avg), does it need the local parameter? Or can it just be def train(color, model, ...)?

# basic training of the model

# Question: train gets model, device, optimizer from initialize_training, which is specified within train function,
# why do I need to call it again before executing the function? Because in vantage6 when I sent the tasks I cannot define that but only in the master function


def RPC_train(data, log_interval, local_dp, epoch, round, delta=1e-5):
    """
    Training the model on all batches.
    Args:
        epoch: The number of the epoch the training is in.
        round: The number of the round the training is in.
        local_dp: Training with local DP?
        delta: The delta value of DP to aim for (default: 1e-5).
    """
    # loading arguments/parameters from first RPC_method
    device, model, optimizer = RPC_initialize_training(data, gamma, learning_rate,
                                                       local_dp)  # is this allowed in vantage6? calling one RPC_method in another?

    train_loader = torch.load(
        "/Users/simontokloth/PycharmProjects/torch-vantage6/v6-ppsdg-py/local/MNIST/processed/training.pt")

    #     train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data',
    #                                                           download=True,
    #                                                             train=True,
    #                                                           transform=transforms.Compose([
    #                                                               transforms.ToTensor(), # first, convert image to PyTorch tensor
    #                                                               transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    #                                                           ])),
    #                                            batch_size=10,
    #                                            shuffle=True)

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # define batch
        batch = (data, target)
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
        # Update the model weights
        optimizer.step()
        #         return loss
        print(loss)

    #         if batch_idx % log_interval == 0:
    #             print('\033[0;{};49m Train on Rank {}, Round {}, Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #             round, epoch, batch_idx * len(batch[0]), len(train_loader.dataset),
    #             100. * batch_idx / len(train_loader), loss.item()))

    # Adding differential privacy or not
    if local_dp == True:
        epsilon, alpha = optimizer.privacy_engine.get_privacy_spent(delta)
        # print("\033[0;{};49m Epsilon {}, best alpha {}".format(epsilon, alpha))


# Model Evaluation

def RPC_test(data):
    """
    Tests the model.

    Args:
        color: The color for the terminal output for this worker.
        model: The model to test.
        device: The device to test the model on.
        test_loader: The local loader for test local. -> no inside function
    """


#     test_loader = torch.load("./testing.pt")
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data',
                                                          download=True,
                                                              train=False,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                              transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                          ])),
                                           batch_size=10,
                                           shuffle=True)

    device, model, optimizer = RPC_initialize_training(data, gamma, learning_rate, local_dp)

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

    print(test_loss)

    # print('\033[0;{};49m \nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #         test_loss, correct, len(test_loader.dataset),
    #         100. * correct / len(test_loader.dataset)))


#-----FED_AVG------

# TODO federated averaging:

# FedAvg gathering of parameters

def RPC_get_parameters(data, model, parameters):
    """
    Get parameters from nodes
    """
    data_size = len(data) // 3  # number of nodes# size of dataset

    weights = []
    # Gather the data sizes on the server
    tensor_weights = torch.tensor(data_size)
    tensor_weights = tensor_weights[1:]
    # Convert all tensors back to weights
    for tensor in tensor_weights:
        weights.append(tensor.item())

    for parameters in model.parameters():
        return {
            "params": parameters,
        }


"""
this might need to be combined with training, so that train 
returns the parameters or that it at least calls the results of training function
"""


# training with those averaged parameters

def RPC_fed_avg(data, local_dp, model, device, optimizer, epoch, parameters, delta=1e-5):
    """
    Training and testing the model on the workers concurrently using federated
    averaging, which means calculating the average of the local model
    parameters after a number of (local) epochs each training round.

    In vantage6, this method will be the training of the model with the average parameters (weighted)

    parameters will need to be specified in args and take parameters from averaged_parameters

    Returns:
        Returns the final model
    """

    # TODO: local: since we usually just get the parameters, this well be an entire task, therefore, we might need to train for each individually
    model = parameters

    for round in range(1, round + 1):

        for epoch in range(1, epoch + 1):
            # Train the model on the workers again
            RPC_train(data, local_dp, model, device, optimizer, epoch, delta=1e-5)
            # Test the model on the workers
            RPC_test(data, model, device)

        gather_params = model.get_parameters()  # or model.parameters()

        RPC_train(model.RPC_average_parameters_weighted(gather_params))

    return model

    ## OR


#     parameters = RPC_average_parameters_weighted(data, model, parameters, weights) # then uses those parameters for training


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




