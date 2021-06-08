from vantage6.tools.mock_client import ClientMockProtocol
import torch

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


task = client.create_new_task(
    input_={
        'method': 'train_test',
        'kwargs': {
            'test_loader': torch.load("C:\\Users\\simon\\PycharmProjects\\torch-vantage6\\v6-ppsdg-py\\local\\MNIST\\processed\\testing.pt"),
            'log_interval': 10,
            'local_dp': False,
            'epoch': 1,
            'delta':  1e-5
        }
    }, organization_ids=org_ids)
print(task)

results = client.get_results(task_id=task.get("id"))


master_task = client.create_new_task({"master": 1, "method":"master"}, [ids[0]])
results = client.get_results(task.get("id"))
print(results)
