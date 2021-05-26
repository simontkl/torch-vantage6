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


    parser.add_argument('-uc', '--use-cuda', action='store_true', default=False,
                    help='enables CUDA (default: false)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01,
                    help='the learning rate (default: 0.01)')
    parser.add_argument('-g', '--gamma', type=float, default=0.7,
                    help='Learning rate step gamma (default: 0.7)')
    # parser.add_argument('-li', '--log-interval', type=int, default=10,
    #                 help='how many batches before logging status (default: 10)')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                    help='the number of epochs to train before aggregation (default: 1)')
    parser.add_argument('-nr', '--num-rounds', type=int, default=1,
                    help='the number of training rounds (default:1)')
    parser.add_argument('-s', '--seed', type=int, help='value for' +
                    ' manual seed')
    # parser.add_argument('-ta', '--train-all', action='store_true', default=False,
    #                 help='trains on all the workers each round, but averages on' +
    #                 ' a subset of parameters instead of only train on subset' +
    #                 ' (default: false)')
    parser.add_argument('-sm', '--save-model', action='store_true', default=False,
                    help='saving model (default: false)')
    parser.add_argument('-ldp', '--local_dp', action='store_true', default=False,
                    help="train with local differential privacy? (default: false)")

    args = parser.parse_args()
    print("\033[0;97;49m Using CUDA for training if available?", args.use_cuda)
    print("\033[0;97;49m Learning rate set to", args.learning_rate)
    print("\033[0;97;49m Gamma set to", args.gamma)
    # print("\033[0;97;49m Log interval set to", args.log_interval)
    print("\033[0;97;49m Epochs set to", args.epochs)
    print("\033[0;97;49m Number of training rounds set to", args.num_rounds)
    print("\033[0;97;49m Manual seed set to", args.seed)
    # print("\033[0;97;49m Training on all workers?", args.train_all)
    print("\033[0;97;49m Saving the model?", args.save_model)
    print("\033[0;97;49m Training with local DP?", args.local_dp)
    print("\033[0;97;49m Finished parsing arguments!")
    print("\033[0;97;49m ===================================")
    return args
