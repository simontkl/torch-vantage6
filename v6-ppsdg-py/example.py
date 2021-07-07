from vantage6.tools.mock_client import ClientMockProtocol

client = ClientMockProtocol(
    datasets=["./local/mnist_dataset.csv"],
    module="v6-ppsdg-py"
)

organizations = client.get_organizations_in_my_collaboration()
org_ids = ids = [organization["id"] for organization in organizations]


master_task = client.create_new_task({
    "master": 1,
    "method": "master"
    },
    [org_ids[0]]
)
results = client.get_results(master_task.get("id"))
print(results)
