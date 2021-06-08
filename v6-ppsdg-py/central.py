"""
Author: Simon Tokloth
Date:
Description: The central algorithm receives the parameters from the nodes and divides by number of nodes-1 for mean parameters.
"""

import torch
import torch.optim as optim
from .v6simplemodel import Net
from opacus import PrivacyEngine
from .algorithms import RPC_get_parameters


def initialize_training(gamma, learning_rate, local_dp):
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
    model = Net().to(device)

    # intializing optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)

    # adding DP if true
    if local_dp == True:
        privacy_engine = PrivacyEngine(model, batch_size=64,
                                       sample_size=60000, alphas=range(2, 32), noise_multiplier=1.3,
                                       max_grad_norm=1.0, )
        privacy_engine.attach(optimizer)

    # returns device, model, optimizer which will be needed in train and test
    return device, model, optimizer


def average_parameters(data, model):
    """
    Get parameters from nodes and calculate the average
    :param model: torch model
    :param parameters: parameters of model
    :param weights:
    :return:
    """

    parameters = RPC_get_parameters(data,
                                    model)  # makes returned parameters from RPC_get_parameters the parameters used in this function

    # TODO: local: since we usually just get the parameters, this well be an entire task, therefore, we might need to train for each individually

    i = 0
    with torch.no_grad():
        for param in parameters:
            s = sum(parameters[i][1:])
            averaged = s / len(parameters)
            param.data = averaged
            i = i + 1
            return {
                "params_averaged": averaged
            }
