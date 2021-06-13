"""
Author: Simon Tokloth
Date:
Description: This module contains the master function which is responsible for the communication.
"""

import time
import torch
import torch
import torch.optim as optim
from .v6simplemodel import Net
from vantage6.tools.util import info
from opacus import PrivacyEngine
from collections import OrderedDict

def master(client, data):
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

    # Determine the device to train on
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize model and send parameters of server to all workers
    model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    """
    Currently the only way to (de-)activate local_dp is by (un-)commenting the next 4 lines
    """

    # Train without federated averaging
    info('Train_test')
    task = client.create_new_task(
        input_={
            'method': 'train',
            'kwargs': {
                'model': model,
                'parameters': model.parameters(),
                'test_loader': torch.load("C:\\Users\\simon\\PycharmProjects"
                                          "\\torch-vantage6\\v6-ppsdg-py\\local\\MNIST\\processed\\testing.pt"),
                'optimizer': optimizer,
                'device': device,
                'log_interval': 10,
                'local_dp': True, # throws error if epoch 2+ or round 2+
                'return_params': True,
                'epoch': 1,
                'round': 1,
                'delta': 1e-5,
            }
        },        organization_ids=ids
    )

    results = client.get_results(task_id=task.get("id"))

    # averag_param = results.state_dict()
    #
    global_sum = 0
    global_count = 0

    for output in results:
        global_sum += output["params"]
        global_count += len(global_sum)

    averaged_parameters = global_sum/global_count

    # in order to not have the optimizer see the new parameters as a non-leaf tensor, .clone().detach() needs
    # to be applied in order to turn turn "grad_fn=<DivBackward0>" into "grad_fn=True"
    averaged_parameters = [averaged_parameters.clone().detach()]


    '''
    Now we need to wait until all organizations(/nodes) finished
    their partial. We do this by polling the server for results. It is
    also possible to subscribe to a websocket channel to get status
    updates.
    '''

    # info("Waiting for results")
    # task_id = task.get("id")
    # task = client.get_task(task_id)
    # while not task.get("complete"):
    #     task = client.get_task(task_id)
    #     info("Waiting for results")
    #     time.sleep(1)

    # # Once we now the partials are complete, we can collect them.
    # info("Obtaining results")
    # results = client.get_results(task_id=task.get("id"))


    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


    info('Federated averaging w/ averaged_parameters')
    task = client.create_new_task(
        input_={
            'method': 'train',
            'kwargs': {
                'model': model,
                'parameters': averaged_parameters,
                'test_loader': torch.load("C:\\Users\\simon\\PycharmProjects"
                                          "\\torch-vantage6\\v6-ppsdg-py\\local\\MNIST\\processed\\testing.pt"),
                'optimizer': optimizer,
                'device': device,
                'log_interval': 10,
                'local_dp': True,
                'return_params': True,
                'epoch': 1,
                'round': 1,
                'delta': 1e-5,
            }
        },
        organization_ids=ids
    )



