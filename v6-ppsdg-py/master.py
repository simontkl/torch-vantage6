"""
Author: Simon Tokloth
Date:
Description: This module contains the master function which is responsible for the communication.
"""

# TODO: in v6-average-py, the only 'results' that were returned were the average partials;
#   that result was then used to create another dict

import time
import torch
from vantage6.tools.util import info


# Own modules

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

    # Request all participating parties to compute their partial. This
    # will create a new task at the central server for them to pick up.
    # We've used a kwarg but is is also possible to use `args`. Although
    # we prefer kwargs as it is clearer.

    # Train without federated averaging
    info('Train and test')
    task = client.create_new_task(
        input_={
            'method': 'train_test',
            'kwargs': {
                'test_loader': torch.load("C:\\Users\\simon\\PycharmProjects"
                                          "\\torch-vantage6\\v6-ppsdg-py\\local\\MNIST\\processed\\testing.pt"),
                'log_interval': 10,
                'local_dp': False,
                'epoch': 10,
                'delta': 1e-5
            }
        },
        organization_ids=ids
    )
