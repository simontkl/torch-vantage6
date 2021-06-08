"""
Author: Simon Tokloth
Date:
Description: The central algorithm receives the parameters from the nodes and divides by number of nodes-1 for mean parameters.
"""

import torch
import torch.optim as optim
from .v6simplemodel import Net
from opacus import PrivacyEngine

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


def average_parameters(params, organizations):
    """
    Get parameters from nodes and calculate the average
    :param node_output_param: the output of RPC_gather_parameters;
        first entry and will be ignored accordingly.; these are the parameters of the model after training
    :param organisations: the organisations defined in master function
    """

    parameters = []
    n_nodes = len(organizations)  # how many organisations?

    for output in params:
        parameters += output["parameters"]

    return {"params_average": parameters / n_nodes}

    # """
    # for comparison, the next code snippet provides the torch.distributed implementation
    # """

#     i = 0
#     with torch.no_grad():
#     for param in model.parameters():
#     # The first entry of the provided parameters when using dist.gather
#     # method also contains the value from the server, remove that one
#     minus_server = parameters[i][1:]
#     # Calculate the average by summing and dividing by the number of
#     # workers
#     s = sum(minus_server)
#     average = s/len(minus_server)
#     # Change the parameter of the global model to the average
#     param.data = average
#     i = i + 1
