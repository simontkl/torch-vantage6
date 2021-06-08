from vantage6.tools.mock_client import ClientMockProtocol

# Initialize the mock server. The datasets simulate the local datasets from
# the node. In this case we have two parties having two different datasets:
# a.csv and b.csv. The module name needs to be the name of your algorithm
# package. This is the name you specified in `setup.py`, in our case that
# would be v6-average-py.
client = ClientMockProtocol(
    datasets=["./local/MNIST/processed/training.pt", "./local/MNIST/processed/testing.pt"],
    module="v6-ppsdg-py"
)

# to inspect which organization are in your mock client, you can run the
# following
organizations = client.get_organizations_in_my_collaboration()
org_ids = ids = [organization["id"] for organization in organizations]


task = client.create_new_task({'method': 'train_test'}, ids)
print(task)


results = client.get_results(task_id=task.get("id"))



master_task = client.create_new_task({"master": 1, "method":"master"}, [ids[0]])
results = client.get_results(task.get("id"))
print(results)
