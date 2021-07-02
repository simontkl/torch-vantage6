# from vantage6.client import Client
#
# # Configure the server details into the client
# client = Client("145.100.131.7", 5000, "")
#
# # Disable encryption, or supply the private key path the to
# # TODO: toggle between encryption or not
#
#
# # Authenticate using a username and password
# client.authenticate("root", "root")
#
# # collect results from computation task
# client.view_result(id=1)
# # {
# #   'assigned_at': '2021-01-14T13:25:44.333898+00:00',
# #   'started_at': None,
# #   'finished_at': None,
# #   'log': 'logging from algorithm',
# #   'result': None,
# #   'input': None,
# #   'id': 3,
# #   'organization': {
# #     'id': 4,
# #     'link': '/organization/4',
# #     'methods': ['GET', 'PATCH']
# #   },
# #   'task': {'id': 1, 'link': '/task/1', 'methods': ['DELETE', 'GET']}
# # }
#
# # you can also filter the output
# client.view_result(id=1, fields=['result', 'log'])
# # {
# #   'result': ...
# #   'log': ...
# # }


# """ Researcher (polling, no websocket but using central container)
# Example on how the researcher should initialize a task without using
# the central container. This means that the central part of the
# algorithm needs to be executed on the machine of the researcher.
# For simplicity this example also uses polling to obtain the results.
# A more advanced example shows how to obtain the results using websockets
# The researcher has to execute the following steps:
# 1) Authenticate to the central-server
# 2) Prepare the input for the algorithm
# 3) Post a new task to a collaboration on the central server
# 4) Wait for central container to finish (polling)
# 5) Obtain the results
# """
import time
#
from vantage6.client import Client
#
# 1. authenticate to the central server
client = Client(
    host="http://172.17.0.2",
    port=3000,
    path="/api"
)

client.authenticate("root", "admin")
client.setup_encryption(None)
#
# # 2. Prepare input for the dsummary Docker image (algorithm)
input_ = {
    "method": "master",
    "kwargs": {'data_format': 'json'}
}


#
# # 3. post the task to the server
task = client.post_task(
    name="FedAvg",
    image="docker-registry.distributedlearning.ai/v6-ppsdg-py",
    collaboration_id=1,
    organization_ids=[4],  # specify where the central container should run! # 4 is the newly created node with the api key that the node config uses
    input_=input_
)
#
# # 4. poll if central container is finished
task_id = task.get("id")
print(f"task id={task_id}")

task = client.request(f"task/{task_id}")
while not task.get("complete"):
    task = client.request(f"task/{task_id}")
    print("Waiting for results...")
    time.sleep(1)
#
# # 5. obtain the finished results
results = client.get_results(task_id=task.get("id"))
#
# # e.g. print the results per node
for result in results:
    node_id = result.get("node")
    print("-"*80)
    print(f"Results from node = {node_id}")
    print(result.get("result"))