import torch
from .algorithms import RPC_train, RPC_test
from .algorithms import RPC_get_parameters



def average_parameters(data, model):
    """
    Get parameters from nodes and calculate the average
    :param model: torch model
    :param parameters: parameters of model
    :param weights:
    :return:
    """

    parameters = RPC_get_parameters()  # makes returned parameters from RPC_get_parameters the parameters used in this function

    # TODO: local: since we usually just get the parameters, this well be an entire task, therefore, we might need to train for each individually

    with torch.no_grad():
        for parameters in model.parameters():
            average = sum(x * y for x, y in zip(parameters[i], weights)) / sum(weights)
            parameters.data = average
            i = i + 1
        return {
            "params_averaged": model
        }


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
