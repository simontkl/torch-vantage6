"""
Author: Simon Tokloth
Date:
Description: This module contains the client that runs the tasks, either individually or together as master_task.
"""

from vantage6.tools.mock_client import ClientMockProtocol

# Initialize the mock server. The datasets simulate the local datasets from
# the node. In this case we have two parties having two different datasets:
# a.csv and b.csv. The module name needs to be the name of your algorithm
# package. This is the name you specified in `setup.py`, in our case that
# would be v6-average-py.
client = ClientMockProtocol(
    datasets=["./local/MNIST/processed/training.pt"],
    module="v6-ppsdg-py"
)

# to inspect which organization are in your mock client, you can run the
# following
organizations = client.get_organizations_in_my_collaboration()
org_ids = ids = [organization["id"] for organization in organizations]

# master task that executes all RPC_methods
# (can also test individually by calling each RPC_method as a task w/o master_task
master_task = client.create_new_task({"master": 1, "method": "master"}, [org_ids[0]])
results = client.get_results(master_task.get("id"))
print(results)
