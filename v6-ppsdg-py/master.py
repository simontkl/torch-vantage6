"""
Author: Simon Tokloth
Date:
Description: This module contains the master function which is responsible for the communication.
"""

import torch
import time
from vantage6.tools.util import info

# Own modules
from .central import average_parameters


def master(client, data, *args, **kwargs): #central algorithm uses the methods of node_algorithm
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

    # Request all participating parties to compute their partial. This
    # will create a new task at the central server for them to pick up.
    # We've used a kwarg but is is also possible to use `args`. Although
    # we prefer kwargs as it is clearer.
    info('Initialising training')
    task = client.create_new_task(
        input_={
            'method': 'initialize_training',
            'kwargs': {

            }
        },
        organization_ids=ids
    )

    ## Train without federated averaging
    info('Train')
    task = client.create_new_task(
        input_={
            'method': 'train',
            'kwargs': {

            }
        },
        organization_ids=ids
    )

    ## Evaluate non-averaged model
    info('Test')
    task = client.create_new_task(
        input_={
            'method': 'test',
            'kwargs': {

            }
        },
        organization_ids=ids
    )

    info('Gather params')
    task = client.create_new_task(
        input_={
            'method': 'get_parameters',
            'kwargs': {

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

    node_output_param = client.get_results(task_id=task.get("id"))

    # calculate the average of the parameters received from model (RPC_get_parameters is executed at each node and should return  those parameters)
    average_parameters(node_output_param, organizations)
    # this function returns a dictionary of the parameters; param = node_output_param, organizations = organizations

    """
    the training happens at the worker nodes. However, as no node-to-node communication is possible, 
    the parameters which are returned by the RPC_get_parameters function. The averaged parameters are then returned 
    and used in the federated_averaging method at the workers with the new parameters
    """

    info('Federated Averaging training')
    task = client.create_new_task(
        input_={
            'method': 'federated_averaging',
            'kwargs': {

            }
        },
        organization_ids=ids
    )


    # Once we now the partials are complete, we can collect them.
    info("Obtaining results")
    results = client.get_results(task_id=task.get("id"))

    info("Master algorithm(s) complete")

    # return all the messages from the nodes
    return results


# TODO We'll need one client.create_new_task for each iteration of the FedAvg