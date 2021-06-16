"""
Author: Simon Tokloth
Date:
Description: The central algorithm receives the parameters from the nodes and divides by number of nodes-1 for mean parameters.
"""

import torch
import torch.optim as optim
from opacus import PrivacyEngine

# Own modules
from .v6simplemodel import Net


def initialize_training(parameters, learning_rate, local_dp):
    """
    Initializes the model, optimizer and scheduler and shares the parameters
    with all the workers in the group.
    This should be sent from server to all nodes.
    Args:
        learning_rate: The learning rate for training.
        local_dp: bool whether to apply local_dp or not.
    Returns:
        Returns the device, model, optimizer and scheduler.
    """

    # Determine the device to train on
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize model and send parameters of server to all workers
    model = Net().to(device)

    # initializing optimizer and scheduler
    optimizer = optim.SGD(parameters, learning_rate, momentum=0.5)

    if local_dp:
        privacy_engine = PrivacyEngine(model, batch_size=64,
                                        sample_size=60000, alphas=range(2, 32), noise_multiplier=1.3,
                                        max_grad_norm=1.0, )
        privacy_engine.attach(optimizer)

    # returns device, model, optimizer which will be needed in train and test
    return device, optimizer, model
