import torch
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine

# Own modules
import v6simplemodel as sm

# ----NODE-----
# RPC_methods always need to start with data
# since parameters of fed_Avg are node-dependent, it should happen in a RPC call; also: everything that is dependeant on data should happen in RPC_call
# if don't want to use data in RPC call: RPC_init_training(_, rank, ...)
maybe

def RPC_initialize_training(data, rank, group, color, args):
    """
    Initializes the model, optimizer and scheduler and shares the parameters
    with all the workers in the group.

    This should be sent from server to all nodes.

    Args:
        rank: The id of the process.
        group: The group the process belongs to.
        color: The color for the terminal output for this worker.
        learning_rate: The learning rate for training.
        cuda: Should we use CUDA?

    Returns:
        Returns the device, model, optimizer and scheduler.
    """
    # Determine the device to train on
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("\033[0;{};49m Rank {} is training on {}".format(color, rank, device))

    # Initialize model and send parameters of server to all workers
    model = sm.Net()
    model.to(device)

    # load data

    # use Opacus for DP: Opacus is a library that enables training PyTorch models
    # with differential privacy. Taken from: https://github.com/pytorch/opacus

    # Intializing optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    if args.local_dp:
        privacy_engine = PrivacyEngine(model, batch_size=64,
            sample_size=60000, alphas=range(2,32), noise_multiplier=1.3,
            max_grad_norm=1.0,)
        privacy_engine.attach(optimizer)

    return device, model, optimizer


def train(data, rank, color, log_interval, model, device, train_loader, optimizer,
    epoch, round, local_dp, delta=1e-5):
    """
    Training the model on all batches.
    Args:
        rank: The id of the process.
        color: The color for the terminal output for this worker.
        log_interval: The amount of rounds before logging intermediate loss.
        model: A model to run training on.
        device: The device to run training on.
        train_loader: Data loader for training data.
        optimizer: Optimization algorithm used for training.
        epoch: The number of the epoch the training is in.
        round: The number of the round the training is in.
        local_dp: Training with local DP?
        delta: The delta value of DP to aim for (default: 1e-5).
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Calculate the loss
        batch = (data, target)
        loss = RPC_train_batch(model, device, batch, optimizer)
        # Log information once every log interval
        if batch_idx % log_interval == 0:
            print('\033[0;{};49m Train on Rank {}, Round {}, Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                color, rank, round, epoch, batch_idx * len(batch[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    if local_dp:
        epsilon, alpha = optimizer.privacy_engine.get_privacy_spent(delta)
        print("\033[0;{};49m Epsilon {}, best alpha {}".format(color, epsilon, alpha))

# train batch is used for SGD
def RPC_train_batch(data, model, device, batch, optimizer, train=True):
    """
    Training the model on one batch of data.

    Args:
        model: A model to run training on.
        device: The device to run training on.
        batch: The batch to train the model on.
        optimizer: Optimization algorithm used for training.
        train: Should we update the model parameters? (default:true)

    Returns:
        The calculated loss after training.
    """
    data, target = batch
    # Send the data and target to the device (cpu/gpu) the model is at
    #line 146 I mean that it is either send to the cpu or to the gpu, but the data is already on the worker node, so don’t interpret that as sending it from the server to the worker
    data, target = data.to(device), target.to(device)
    # Clear gradient buffers
    optimizer.zero_grad()
    # Run the model on the data
    output = model(data)
    # Calculate the loss
    loss = F.nll_loss(output, target)
    # Calculate the gradients
    loss.backward()

    # Update the model weights
    if train:
        optimizer.step()
    return loss


def RPC_test(data, rank, color, model, device, test_loader):
    """
    Tests the model.

    Args:
        rank: The id of the process.
        color: The color for the terminal output for this worker.
        model: The model to test.
        device: The device to test the model on.
        test_loader: The data loader for test data.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Send the data and target to the device (cpu/gpu) the model is at
            data, target = data.to(device), target.to(device)
            # Run the model on the data
            output = model(data)
            # Calculate the loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # Check whether prediction was correct
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\033[0;{};49m \nTest set on Rank {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        color, rank, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


#-----FED_AVG------

# TODO federated averaging:

# def RPC_get_parameters(data, client, node):
#     """
#     Get parameters from nodes
#     """
#
# "for parameters in nodes:
#       return parameters"


def RPC_average_parameters_weighted(data, model, parameters, weights):
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

def RPC_fed_avg(data, args, model, optimizer, train_loader, test_loader, device):
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
