"""
Author: Simon Tokloth
Date:
Description: This module contains the master function which is responsible for the communication.
"""

import time
import torch
from vantage6.tools.util import info

# Own modules

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


    """
    return the values and use them as arguments for train 
    """


    learning_rate = 0.01

    ## Train without federated averaging
    info('Train and test')
    task = client.create_new_task(
        input_={
            'method': 'train_test',
            'kwargs': {
                'data2': 'test_loader',
                'log_interval': 10,
                'local_dp': False,
                'epoch': 1,
                'delta': 1e-5
            }
        },
        organization_ids=ids
    )



    info("Waiting for results")
    task_id = task.get("id")
    task = client.get_task(task_id)
    while not task.get("complete"):
        task = client.get_task(task_id)
        info("Waiting for results")
        time.sleep(1)

    info("Obtaining results")
    results = client.get_results(task_id=task.get("id"))

    info("Master algorithm(s) complete")

    # return all the messages from the nodes
    return results


