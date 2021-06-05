"""
Author: Simon Tokloth
Date:
Description: The central algorithm receives the parameters from the nodes and divides by number of nodes-1 for mean parameters.
"""

import torch
from .algorithms import RPC_train, RPC_test
from .algorithms import RPC_get_parameters


def average_parameters(node_output_param, organisations):
    """
    Get parameters from nodes and calculate the average
    :param node_output_param: the output of RPC_gather_parameters
    :param organisations: the organisations defined in master function
    """

    parameters = []
    n_nodes = len(organisations)  # how many organisations?

    for output in node_output_param:
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
