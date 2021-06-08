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
from opacus import PrivacyEngine
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

    # Determine the device to train on
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize model and send parameters of server to all workers
    model = Net()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # if local_dp:
    #     privacy_engine = PrivacyEngine(model, batch_size=64,
    #                                    sample_size=60000, alphas=range(2, 32), noise_multiplier=1.3,
    #                                    max_grad_norm=1.0, )
    #     privacy_engine.attach(optimizer)

    # Train without federated averaging
    info('Train_test')
    task = client.create_new_task(
        input_={
            'method': 'train_test',
            'kwargs': {
                'model': model,
                'parameters': model.parameters(),
                'test_loader': torch.load("C:\\Users\\simon\\PycharmProjects"
                                          "\\torch-vantage6\\v6-ppsdg-py\\local\\MNIST\\processed\\testing.pt"),
                'optimizer': optimizer,
                'device': device,
                'log_interval': 10,
                'local_dp': False,
                'epoch': 1,
                'round': 1,
                'delta': 1e-5,
                'optim': True
            }
        },
        organization_ids=ids
    )

    trained_model = torch.load(
        'C:\\Users\\simon\\PycharmProjects\\torch-vantage6\\v6-ppsdg-py\\local\\model_trained.pth')

    info('Gather params')
    task = client.create_new_task(
        input_={
            'method': 'get_parameters',
            'kwargs': {
                'model': trained_model
            }
        },
        organization_ids=ids
    )

    '''
    Now we need to wait until all organizations(/nodes) finished
    their partial. We do this by polling the server for results. It is
    also possible to subscribe to a websocket channel to get status
    updates.
    '''

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

    # averaging of returned parameters
    global_sum = 0
    global_count = 0

    for output in results:
        global_sum += output["params"]
        global_count += len(global_sum)

    averaged_parameters = (global_sum/global_count) / len(organizations)  # in testing this would probably mean / 1

    new_params = {'averaged_parameters': averaged_parameters}

    info('Federated averaging w/ averaged_parameters')
    task = client.create_new_task(
        input_={
            'method': 'train_test',
            'kwargs': {
                'model': model,
                'parameters': new_params,
                'test_loader': torch.load("C:\\Users\\simon\\PycharmProjects"
                                          "\\torch-vantage6\\v6-ppsdg-py\\local\\MNIST\\processed\\testing.pt"),
                'optimizer': optimizer,
                'device': device,
                'log_interval': 10,
                'local_dp': False,
                'epoch': 1,
                'round': 1,
                'delta': 1e-5,
                'optim': False
            }
        },
        organization_ids=ids
    )

