# with the vantage6 implementation of the algorithm it is time to test it.
# Before we run it into a vantage6 setup we can test it locally by using the
# ClientMockProtocol which simulates the communication with the central server.
# this is the testing of the functionality of the implementation
# mock client has the same functions as client
# in https://github.com/IKNL/vantage6-client/blob/master/vantage6/client/__init__.py
#
# non-mock implementation

# from vantage6.client import Client
# from pathlib import Path
#
# # Create, athenticate and setup client
# client = Client("http://127.0.0.1", 5000, "")
# client.authenticate("frank@iknl.nl", "password")
# client.setup_encryption(None)
#
# # Define algorithm input
# input_ = {
#     "master": "true",
#     "method":"master",
#     "args": [
#       {
#         "num_awards":"Int64",
#         "prog":"category", "math":"Int64"
#       }
#     ],
#     "kwargs": {}
# }
#
# # Send the task to the central server
# task = client.post_task(
#     name="testing",
#     image="harbor.vantage6.ai/algorithms/summary",
#     collaboration_id=1,
#     input_= input_,
#     organization_ids=[2]
# )
#
# # Retrieve the results
# res = client.get_results(task_id=task.get("id"))

from vantage6.tools.mock_client import ClientMockProtocol

# Initialize the mock server. The datasets simulate the local datasets from
# the node. In this case we have two parties having two different datasets:
# a.csv and b.csv. The module name needs to be the name of your algorithm
# package. This is the name you specified in `setup.py`, in our case that
# would be v6-average-py.
client = ClientMockProtocol(
    datasets=["./local/mnist_train.csv", "./local/mnist_test.csv"],
    module="v6-ppsdg-py"
)

# to inspect which organization are in your mock client, you can run the
# following
organizations = client.get_organizations_in_my_collaboration()
org_ids = ids = [organization["id"] for organization in organizations]


task = client.create_new_task({'method': 'initialize_training'}, ids)
print(task)


task = client.create_new_task({'method': 'train'}, ids)
print(task)


task = client.create_new_task({'method': 'test'}, ids)
print(task)


task = client.create_new_task({'method': 'get_parameters'}, ids)
print(task)


# node_output_param = client.get_results(task_id=task.get("id")) # do I need to specify this here?


task = client.create_new_task({'method': 'federated_averaging'}, ids)
print(task)


results = client.get_results(task_id=task.get("id"))



master_task = client.create_new_task({"master": 1, "method":"master"}, [ids[0]])
results = client.get_results(task.get("id"))
print(results)
