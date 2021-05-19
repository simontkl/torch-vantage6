import torch
import time
from vantage6.tools.util import info

# Own modules
import parser as parser



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
    info('Requesting partial computation')
    task = client.create_new_task(
        input_={
            'method': 'average_partial',
            'kwargs': {

            }
        },
        organization_ids=ids
    )

    # Now we need to wait until all organizations(/nodes) finished
    # their partial. We do this by polling the server for results. It is
    # also possible to subscribe to a websocket channel to get status
    # updates.
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

parser.parse_arguments()

# TODO federated averaging:

# def get_parameters(client, node):
#     """
#     Get parameters from nodes
#     """
#
# "for parameters in nodes:
#       return parameters"


def average_parameters_weighted(model, parameters, weights):
    """
    Get parameters from nodes and calculate the average
    :param model: torch model
    :param parameters: parameters of model
    :param weights:
    :return:
    """
    with torch.no_grad():
        for parameters in model.parameters():
            average = sum(x * y for x, y in zip(parameters[i], weights)) / sum(weights)
            parameters.data = average
            i = i + 1
        return parameters

def fed_avg(args, model, optimizer, train_loader, test_loader, device):
    """
    Training and testing the model on the workers concurrently using federated
    averaging, which means calculating the average of the local model
    parameters after a number of (local) epochs each training round.


    Returns:
        Returns the final model
    """

    for epoch in range(1, args.epochs + 1):
        # Train the model on the workers
        model.train(args.log_interval, model, device, train_loader,
              optimizer, epoch, round, args.local_dp)
        # Test the model on the workers
        model.test(model, device, test_loader)

    gather_params = model.get_parameters()

    model.average_parameters_weighted(gather_params)

    return model

# TODO DATA !! -> send to nodes full dataset or sample and do indexing at node
