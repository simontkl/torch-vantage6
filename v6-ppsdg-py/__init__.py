"""
Author: Simon Tokloth
Date:
Description: This module contains the master function which is responsible for the communication.
"""

import time
from vantage6.tools.util import info


def master(client, data, ids):
    """Master algorithm.
    The master algorithm is the chair of the Round Robin, which makes
    sure everyone waits for their turn to identify themselves.
    """
    # Info messages can help you when an algorithm crashes. These info
    # messages are stored in a log file which is send to the server when
    # either a task finished or crashes.

    start_time = time.time()

    info('Collecting participating organizations')

    # Collect all organization that participate in this collaboration.
    # These organizations will receive the task to compute the partial.
    # organizations = client.get_organizations_in_my_collaboration()
    # ids = [organization.get("id") for organization in organizations]
    # print(organizations)

    # # Determine the device to train on
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") #"cuda" if use_cuda else "cpu"

    # clear cuda memory
    torch.cuda.empty_cache()
    # torch.cuda.clear_memory_allocated()

    # # Initialize model and send parameters of server to all workers
    model = Net().to(device)

    # Train without federated averaging
    info('Train_test')
    task = client.create_new_task(
        input_={
            'method': 'train_test',
            'kwargs': {
                # 'organizations': organizations,
                'parameters': list(model.parameters()),
                'model': model,
                'device': device,
                'log_interval': 10,
                # 'local_dp': True,
                'return_params': True,
                'epoch': 1,
                'if_test': False,
                'ids': ids
            }
        }, organization_ids=[ids]
    )

    info("Waiting for parameters")
    task_id = task.get("id")
    task = client.get_task(task_id)
    while not task.get("complete"):
        task = client.get_task(task_id)
        info("Waiting for results")
        time.sleep(1)

    # # Once we now the partials are complete, we can collect them.
    info("Obtaining parameters from all nodes")

    results_train = client.get_results(task_id=task.get("id"))

    global_sum = 0
    global_count = 0

    for output in results_train:
        # print(len(output))
        global_sum += output["params"]
        global_count += len(global_sum)

    # for parameters in results:
    #     print(parameters)

    averaged_parameters = global_sum / global_count

    # info("Averaged parameters")
    # for parameters in averaged_parameters:
    #     print(parameters)

    """
    in order to not have the optimizer see the new parameters as a non-leaf tensor, .clone().detach() needs
    to be applied in order to turn turn "grad_fn=<DivBackward0>" into "grad_fn=True"
    """

    averaged_parameters = [averaged_parameters.clone().detach()]

    torch.cuda.empty_cache()
    # torch.cuda.clear_memory_allocated()

    # info('Testing first round')
    # task = client.create_new_task(
    #     input_={
    #         'method': 'train_test',
    #         'kwargs': {
    #             'organizations': organizations,
    #             'parameters': averaged_parameters,
    #             'model': output['model'],
    #             'device': device,
    #             'log_interval': 10,
    #             'local_dp': False,
    #             'return_params': True,
    #             'epoch': 5,
    #             # 'round': 1,
    #             'if_test': True,
    #             'cifar': False
    #         }
    #     },
    #     organization_ids=ids
    # )
    #
    # results_test = client.get_results(task_id=task.get("id"))
    #
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    # for output in results_test:
    #     acc = output["test_accuracy"]
    # return 'Accuracy: {:.2f}%'.format(acc)




    # info('Federated averaging w/ averaged_parameters')
    # task = client.create_new_task(
    #     input_={
    #         'method': 'train_test',
    #         'kwargs': {
    #             'organizations': organizations,
    #             'parameters': averaged_parameters,
    #             'model': output['model'],
    #             'device': device,
    #             'log_interval': 10,
    #             'local_dp': False,
    #             'return_params': True,
    #             'epoch': 1,
    #             'if_test': False,
    #             'cifar': False,
    #         }
    #     },
    #     organization_ids=ids
    # )

    info('Federated averaging w/ averaged_parameters')
    task = client.create_new_task(
        input_={
            'method': 'train_test',
            'kwargs': {
                # 'organizations': organizations,
                'parameters': averaged_parameters,
                'model': output['model'],
                'device': device,
                'log_interval': 10,
                # 'local_dp': False,
                'return_params': True,
                'epoch': 1,
                'if_test': True,
                'ids': ids
            }
        },
        organization_ids=[ids]
    )

    results = client.get_results(task_id=task.get("id"))

    print("--- %s seconds ---" % (time.time() - start_time))

    for output in results:
        acc = output["test_accuracy"] # return output["test_accuracy"]

    return 'Accuracy: {:.2f}%'.format(acc) #(acc)

# print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#                 test_loss, correct, len(test_loader.dataset),
#                 100. * correct / len(test_loader.dataset)))
#             test_accuracy = 100. * correct / len(test_loader.dataset)

"""
Author: Simon Tokloth
Date:
Description: This module contains the RPC_methods including the training and federated averaging.
"""

# Import packages
import torch
import torch.nn.functional as F
import torch.optim as optim
# from opacus import PrivacyEngine
from .v6simplemodel import Net
# from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import psutil

# training of the model
def RPC_train_test(data, model, parameters, device, log_interval, return_params, epoch, if_test, ids):
    """
    :param data:
    :param model:
    :param parameters:
    :param device:
    :param log_interval:
    :param local_dp:
    :param return_params:
    :param epoch:
    :param if_test:
    :return:
    """

    train = data
    train_batch_size = 64
    test_batch_size = 64

    # X = train
    X = (train.iloc[:, 1:].values).astype('float32')
    # Y = train
    Y = train.iloc[:, 0].values
    features_train, features_test, targets_train, targets_test = train_test_split(X, Y, test_size=0.2,
                                                                                  random_state=42)
    X_train = torch.from_numpy(features_train / 255.0)
    X_test = torch.from_numpy(features_test / 255.0)

    Y_train = torch.from_numpy(targets_train).type(torch.LongTensor)
    Y_test = torch.from_numpy(targets_test).type(torch.LongTensor)

    train = torch.utils.data.TensorDataset(X_train, Y_train)
    test = torch.utils.data.TensorDataset(X_test, Y_test)

    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size, shuffle=False)

    # if input is train.pt
    # train_loader = data

    learning_rate = 0.01

    # trainloader_cifar = torch.utils.data.DataLoader(trainset, batch_size=4,
    #                                           shuffle=True, num_workers=2)
    # testloader_cifar = torch.utils.data.DataLoader(testset, batch_size=4,
    #                                           shuffle=False, num_workers=2)

    test_accuracy = 0
    if if_test:
        model.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                # Send the local and target to the device (cpu/gpu) the model is at
                data, target = data.to(device), target.to(device)
                # Run the model on the local
                batch_size = data.shape[0]
                # print(batch_size)
                # if cifar:
                #     data = data.reshape(batch_size, 32, 32, 3)
                #     data = data.unsqueeze(0)
                #     data = data.reshape(batch_size, 3, 32, 32)
                #     # data = data.unsqueeze(1)
                # else:
                data = data.reshape(batch_size, 28, 28)
                data = data.unsqueeze(1)
                output = model(data)
                # Calculate the loss
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # Check whether prediction was correct
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
            test_accuracy = 100. * correct / len(test_loader.dataset)

    else:
        learning_rate = 0.01

        # if local_dp == True:
        # initializing optimizer and scheduler
        optimizer = optim.SGD(list(parameters), lr=learning_rate, momentum=0.5)

        # privacy_engine = PrivacyEngine(model, batch_size=64,
        #                                sample_size=60000, alphas=range(2, 32), noise_multiplier=1.3,
        #                                max_grad_norm=1.0, )
        #
        # if local_dp:
        #     privacy_engine.attach(optimizer)

        model.train()
        for epoch in range(1, epoch + 1):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                # Clear gradient buffers
                optimizer.zero_grad()

                batch_size = data.shape[0]
                # print(batch_size)

                # if cifar:
                #     data = data.reshape(batch_size, 32, 32, 3)
                #     data = data.unsqueeze(0)
                #     data = data.reshape(batch_size, 3, 32, 32)
                #     # data = data.unsqueeze(1)
                # else:
                data = data.reshape(batch_size, 28, 28)
                data = data.unsqueeze(1)
                # data = data.reshape(batch_size, 3, 32, 32)
                # data = data.unsqueeze(0)
                # print(data.shape)
                # print(data.type())
                # print(target.type())

                # Run the model on the data
                output = model(data)
                # Calculate the loss
                loss = F.nll_loss(output, target)
                # Calculate the gradients
                loss.backward()
                # Update model
                optimizer.step()

                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()), psutil.cpu_percent(), psutil.virtual_memory().percent)


    if return_params:
        for parameters in list(model.parameters()):  # model.parameters() but should be the same since it's the argument
            return {'params': parameters,
                    'model': model,
                    'test_accuracy': test_accuracy}






