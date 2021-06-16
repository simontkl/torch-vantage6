"""
Author: Simon Tokloth
Date:
Description: This module contains the RPC_methods including the training and federated averaging.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from .v6simplemodel import Net

# Own modules
from .central import initialize_training


# training of the model
def RPC_train(data, parameters, log_interval, local_dp, epoch, delta, return_params):
    """
    Training the model on all batches.
    Args:
        epoch: The number of the epoch the training is in.
        model: use model from initialize_training, if it doesn't work try all separately and change model params in fed_avg
        optimizer:
        device:
        log_interval: The amount of rounds before logging intermediate loss.
        local_dp: Training with local DP?
        epoch
        return_params

        delta: The delta value of DP to aim for (default: 1e-5).
    """
    optimizer = optim.SGD(parameters, lr=0.01, momentum=0.5)

    device, model = initialize_training(optimizer, 0.01, local_dp)
    # optimizer = optim.SGD(parameters, lr=0.01, momentum=0.5)
    #
    # if local_dp:
    #     privacy_engine = PrivacyEngine(model, batch_size=64,
    #                                     sample_size=100, alphas=range(2, 32), noise_multiplier=1.3,
    #                                     max_grad_norm=1.0, )
    #     privacy_engine.attach(optimizer)

    train_data = data

    model.train()
    # for epoch in range(1, epoch +1):
    for batch_idx, (data, target) in enumerate(train_data):
        # Send the data and target to the device (cpu/gpu) the model is at
        data, target = data.to(device), target.to(device)
        # Clear gradient buffers
        optimizer.zero_grad()
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
                    epoch, batch_idx * len(data), len(train_data.dataset),
                    100. * batch_idx / len(train_data), loss.item()))
    if local_dp:
        epsilon, alpha = optimizer.privacy_engine.get_privacy_spent(delta)
        print("\nEpsilon {}, best alpha {}".format(epsilon, alpha))

    torch.save(model.state_dict(), f"C:\\Users\\simon\\PycharmProjects"
                                   f"\\torch-vantage6\\v6-ppsdg-py\\local\\model_trained.pth")

    if return_params:
        for parameters in model.parameters():
            return {'params': parameters}


def RPC_test(data, device):

    test_loader = torch.load("C:\\Users\\simon\\PycharmProjects\\torch-vantage6\\v6-ppsdg-py"
                             "\\local\\MNIST\\processed\\testing.pt")

    model = Net().to(device)
    model_trained = torch.load(f"C:\\Users\\simon\\PycharmProjects"
                               f"\\torch-vantage6\\v6-ppsdg-py\\local\\model_trained.pth")

    model.load_state_dict(model_trained)

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Send the local and target to the device (cpu/gpu) the model is at
            data, target = data.to(device), target.to(device)
            # Run the model on the local
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


# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                           shuffle=False, num_workers=2)

def RPC_train_test(data, model, device, parameters, log_interval, local_dp, epoch, delta, round, return_params):

    for round in range(1, round+1):
        for epoch in range(1, epoch+1):
            RPC_train(data, parameters, log_interval, local_dp, epoch, delta, return_params)
            RPC_test(data, device)