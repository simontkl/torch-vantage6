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
    # print(organizations)

    # # Determine the device to train on
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") #

    # clear cuda memory
    torch.cuda.empty_cache()
    # torch.cuda.clear_memory_allocated()

    # # Initialize model and send parameters of server to all workers
    model = Net().to(device)

    # Train without federated averaging
    info('Train_test')
    task = client.create_new_task(
        input_={
            'method': 'train_test',
            'kwargs': {
                'organizations': organizations,
                'parameters': model.parameters(),
                'model': model,
                'device': device,
                'log_interval': 10,
                'local_dp': True,
                'return_params': True,
                'epoch': 2,
                # 'round': 1,
                'delta': 1e-5,
                'if_test': False
            }
        }, organization_ids=ids
    )

    info("Waiting for parameters")
    task_id = task.get("id")
    task = client.get_task(task_id)
    while not task.get("complete"):
        task = client.get_task(task_id)
        info("Waiting for results")
        time.sleep(1)

    # # Once we now the partials are complete, we can collect them.
    info("Obtaining parameters from all nodes")

    results_train = client.get_results(task_id=task.get("id"))

    global_sum = 0
    global_count = 0

    for output in results_train:
        # print(len(output))
        global_sum += output["params"]
        global_count += len(global_sum)

    # for parameters in results:
    #     print(parameters)

    averaged_parameters = global_sum / global_count

    # info("Averaged parameters")
    # for parameters in averaged_parameters:
    #     print(parameters)

    """
    in order to not have the optimizer see the new parameters as a non-leaf tensor, .clone().detach() needs
    to be applied in order to turn turn "grad_fn=<DivBackward0>" into "grad_fn=True"
    """

    averaged_parameters = [averaged_parameters.clone().detach()]

    torch.cuda.empty_cache()
    # torch.cuda.clear_memory_allocated()

    # info('Federated averaging w/ averaged_parameters')
    # task = client.create_new_task(
    #     input_={
    #         'method': 'train_test',
    #         'kwargs': {
    #             'parameters': averaged_parameters,
    #             'model': output['model'],
    #             'device': device,
    #             'log_interval': 10,
    #             'local_dp': True,
    #             'return_params': True,
    #             'epoch': 5,
    #             # 'round': 1,
    #             'delta': 1e-5,
    #             'if_test': False
    #         }
    #     },
    #     organization_ids=ids
    # )

    info('Federated averaging w/ averaged_parameters')
    task = client.create_new_task(
        input_={
            'method': 'train_test',
            'kwargs': {
                'organizations': organizations,
                'parameters': averaged_parameters,
                'model': output['model'],
                'device': device,
                'log_interval': 10,
                'local_dp': False,
                'return_params': True,
                'epoch': 1,
                # 'round': 1,
                'delta': 1e-5,
                'if_test': False
            }
        },
        organization_ids=ids
    )

    info('Federated averaging w/ averaged_parameters')
    task = client.create_new_task(
        input_={
            'method': 'train_test',
            'kwargs': {
                'organizations': organizations,
                'parameters': averaged_parameters,
                'model': output['model'],
                'device': device,
                'log_interval': 10,
                'local_dp': False,
                'return_params': True,
                'epoch': 1,
                # 'round': 1,
                'delta': 1e-5,
                'if_test': True
            }
        },
        organization_ids=ids
    )

    results = client.get_results(task_id=task.get("id"))
    for output in results:
        acc = output["test_accuracy"]
    return acc