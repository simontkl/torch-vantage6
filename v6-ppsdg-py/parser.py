"""
Author: Carlijn Nijhuis
Date: 13-01-21
Description: This module contains functions for parsing arguments.
"""

import argparse

def parse_arguments():
    """
    Parses the arguments used by main.py.

    Returns:
        The parsed arguments.
    """
    print("\033[0;97;49m ===================================")
    print("\033[0;97;49m Parsing arguments...")
    parser = argparse.ArgumentParser()

    parser.add_argument('-nn', '--num-nodes', type=int, default=2,
                    help='the number of nodes to create, including server ' +
                    '(default: 2)')
    parser.add_argument('-uc', '--use-cuda', action='store_true', default=False,
                    help='enables CUDA (default: false)')
    parser.add_argument('-bs', '--batch-size', type=int, default=64,
                    help='the input batch size for training (default: 64)')
    parser.add_argument('-tbs', '--test-batch-size', type=int, default=1000,
                    help='the input batch size for testing (default: 1000)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01,
                    help='the learning rate (default: 0.01)')
    parser.add_argument('-g', '--gamma', type=float, default=0.7,
                    help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('-li', '--log-interval', type=int, default=10,
                    help='how many batches before logging status (default: 10)')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                    help='the number of epochs to train before aggregation (default: 1)')
    parser.add_argument('-nr', '--num-rounds', type=int, default=1,
                    help='the number of training rounds (default:1)')
    parser.add_argument('-s', '--seed', type=int, help='value for' +
                    ' manual seed')
    parser.add_argument('-gs', '--group-size', type=int, default=1,
                    help='the size of the subgroups determined each training ' +
                    'round, excluding server (default: 1)')
    parser.add_argument('-ta', '--train-all', action='store_true', default=False,
                    help='trains on all the workers each round, but averages on' +
                    ' a subset of parameters instead of only train on subset' +
                    ' (default: false)')
    parser.add_argument('-sm', '--save-model', action='store_true', default=False,
                    help='saving model (default: false)')
    parser.add_argument('-sgd', '--stochastic-gradient', action='store_true',
                    default=False, help='running stochastic gradient descent (default: false)')
    parser.add_argument('-gp', '--gradient-penalty', type=int, default=10,
                    help='the loss weight for the gradient penalty (default:10)')
    parser.add_argument('-gan', '--gan', action='store_true', default=False,
                    help='train a GAN and not a classifier (default: false)')
    parser.add_argument('-ldp', '--local_dp', action='store_true', default=False,
                    help="train with local differential privacy? (default: false)")

    args = parser.parse_args()
    print("\033[0;97;49m Number of nodes set to", args.num_nodes)
    print("\033[0;97;49m Using CUDA for training if available?", args.use_cuda)
    print("\033[0;97;49m Batch size set to", args.batch_size)
    print("\033[0;97;49m Test batch size set to", args.test_batch_size)
    print("\033[0;97;49m Learning rate set to", args.learning_rate)
    print("\033[0;97;49m Gamma set to", args.gamma)
    print("\033[0;97;49m Log interval set to", args.log_interval)
    print("\033[0;97;49m Epochs set to", args.epochs)
    print("\033[0;97;49m Number of training rounds set to", args.num_rounds)
    print("\033[0;97;49m Manual seed set to", args.seed)
    print("\033[0;97;49m Group size set to", args.group_size)
    print("\033[0;97;49m Training on all workers?", args.train_all)
    print("\033[0;97;49m Saving the model?", args.save_model)
    print("\033[0;97;49m Running FedSGD?", args.stochastic_gradient)
    print("\033[0;97;49m Loss weight for gradient penalty set to", args.gradient_penalty)
    print("\033[0;97;49m Training a GAN?", args.gan)
    print("\033[0;97;49m Training with local DP?", args.local_dp)
    print("\033[0;97;49m Finished parsing arguments!")
    print("\033[0;97;49m ===================================")
    return args
