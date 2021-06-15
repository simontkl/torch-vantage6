"""
Author: Simon Tokloth
Date:
Description: This module contains the master function which is responsible for the communication.
"""

import time
import torch
from .v6simplemodel import Net
from vantage6.tools.util import info


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

    # # Determine the device to train on
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # # Initialize model and send parameters of server to all workers
    model = Net().to(device)

    # Train without federated averaging
    info('Train_test')
    task = client.create_new_task(
        input_={
            'method': 'train',
            'kwargs': {
                'model': model,
                'parameters': model.parameters(),
                'device': device,
                'log_interval': 10,
                'local_dp': False, # throws error if epoch 2+ or round 2+
                'return_params': True,
                'epoch': 1,
                'round': 1,
                'delta': 1e-5,
            }
        },        organization_ids=ids
    )

    '''
    Now we need to wait until all organizations(/nodes) finished
    their partial. We do this by polling the server for results. It is
    also possible to subscribe to a websocket channel to get status
    updates.
    '''

    info("Waiting for parameters")
    task_id = task.get("id")
    task = client.get_task(task_id)
    while not task.get("complete"):
        task = client.get_task(task_id)
        info("Waiting for results")
        time.sleep(1)

    # # Once we now the partials are complete, we can collect them.
    info("Obtaining parameters from all nodes")
    
    results = client.get_results(task_id=task.get("id"))

    global_sum = 0
    global_count = 0

    for output in results:
        global_sum += output["params"]
        global_count = len(global_sum)

    averaged_parameters = global_sum/global_count # same as len(organizations)

    # in order to not have the optimizer see the new parameters as a non-leaf tensor, .clone().detach() needs
    # to be applied in order to turn turn "grad_fn=<DivBackward0>" into "grad_fn=True"
    averaged_parameters = [averaged_parameters.clone().detach()]

    info('Federated averaging w/ averaged_parameters')
    task = client.create_new_task(
        input_={
            'method': 'train',
            'kwargs': {
                'model': model,
                'parameters': averaged_parameters,
                'device': device,
                'log_interval': 10,
                'local_dp': False,
                'return_params': True,
                'epoch': 1,
                'round': 1,
                'delta': 1e-5,
            }
        },
        organization_ids=ids
    )



