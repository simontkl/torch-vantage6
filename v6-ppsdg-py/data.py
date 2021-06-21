import os, os.path
import sys
import copy
from itertools import cycle
import pandas as pd
import numpy as np
import random
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision import datasets, transforms
import prettytable as pt
import data as dat


# Load MNIST dataset from torchvision - train set (60000 samples) and test set (10000 samples)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set_pre = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set_pre = datasets.MNIST('./data', train=False, transform=transform)
# Merge train set and test set as the whole MNIST dataset (70000 samples)
dataset_mnist_all = dat.MergeDatasets([train_set_pre, test_set_pre])

# Shuffle the whole MNIST dataset (70000 samples)
dataset_mnist_shuffled = dat.ShuffleDataset(dataset_mnist_all)
# Save the shuffled whole MNIST dataset
torch.save(dataset_mnist_shuffled, './mnist_shuffled.pt')

def CreateMapLabelIndexes(dataset):
    label_indexes_map = {}
    label_cnt_map = {}
    labels = dataset.targets.unique()
    for label in labels:
        label_indexes = (dataset.targets==label.item()).nonzero(as_tuple=True)[0]
        label_indexes_map[label.item()] = label_indexes
        label_cnt_map[label.item()] = len(label_indexes)
    return label_indexes_map, label_cnt_map

def GetPartitionedDataset(dataset, df_indexes_per_class, partition_index):
    valid_idx= torch.cat(list(df_indexes_per_class.loc[partition_index]))

    idx = torch.tensor([False]*len(dataset.targets))
    idx[valid_idx] = True

    dataset_partition = copy.deepcopy(dataset)
    # show_samples_per_class(dataset_partition)
    # Only get the labels in the entries with the correct indexes
    dataset_partition.targets = dataset_partition.targets[idx]
    # Only get the data in the entries with the correct indexes
    dataset_partition.data = dataset_partition.data[idx]

    return dataset_partition

# Get a specific partition of the dataset, given the corresponding indexes for all worker nodes
# def get_partitioned_dataset(dataset, df_indexes_per_class, partition_index):
def GetPartitionedDataset(dataset, df_indexes_per_class, partition_index):
    valid_idx= torch.cat(list(df_indexes_per_class.loc[partition_index]))

    idx = torch.tensor([False]*len(dataset.targets))
    idx[valid_idx] = True

    dataset_partition = copy.deepcopy(dataset)
    # show_samples_per_class(dataset_partition)
    # Only get the labels in the entries with the correct indexes
    dataset_partition.targets = dataset_partition.targets[idx]
    # Only get the data in the entries with the correct indexes
    dataset_partition.data = dataset_partition.data[idx]

    return dataset_partition

# Merge a list of same-type dataset (eg: merge 3 MNIST-type datasets)
def MergeDatasets(dataset_list):
    # Exit if there is no datasets to be merged
    if (len(dataset_list)==0):
        print("No datasets to be merged")
        sys.exit(1)
    else:
        pass
    merged_dataset = copy.deepcopy(dataset_list[0])
    merged_dataset.targets = torch.cat(list(x.targets for x in dataset_list))
    merged_dataset.data = torch.cat(list(x.data for x in dataset_list))
    # merged dataset will have different orders for samples (like combining several lists)
    # eg: [1,34,67] + [23,53,42] = [1,34,67,23,53,42]
    # but data and indexes for the samples are the same
    return merged_dataset

# Merge index of samples for a dataset (eg: shuffle full MNIST dataset = 60000 training samples + 10000 testing samples)
def ShuffleDataset(dataset):
    # Exit if there is the dataset is empty
    if (len(dataset)==0):
        sys.exit(1)
    else:
        pass
    shuffled_dataset = copy.deepcopy(dataset)
    indices = list(range(len(dataset)))
    # np.random.seed(10242048)
    # np.random.seed()
    np.random.shuffle(indices)

    shuffled_dataset.targets = shuffled_dataset.targets[indices]
    shuffled_dataset.data = shuffled_dataset.data[indices]

    return shuffled_dataset

def data_dist_FullyIID_each(dataset, n_workers):
    partition_pcts = [1/n_workers]*n_workers

    _, df_dist_fullyIID_cnt, df_dist_fullyIID_indexes = EqualPartitionEachClass(dataset, partition_pcts)

    return df_dist_fullyIID_cnt, df_dist_fullyIID_indexes


def EqualPartitionEachClass(dataset, partition_pcts, return_part_index=1):
    labels, label_samples_cnt = dataset.targets.unique(return_counts=True)
    label_indexes_map, label_cnt_map = CreateMapLabelIndexes(dataset)

# Build a dataframe containing #samples per class in each node. Rows are partition_index, columns are class labels.
    # df_samples_cnt_per_class = pd.DataFrame(index=range(len(partition_pcts)-1), columns=labels.tolist())
    df_samples_cnt_per_class = pd.DataFrame(index=range(1,len(partition_pcts)), columns=labels.tolist())
    for x in range(1,len(partition_pcts)):
        partition_pct = partition_pcts[x]
        # print("partition {}: {} pct".format(str(x),partition_pct))
        samples_cnt = [int(partition_pct*label_cnt_map[x]) for x in label_cnt_map]
        df_samples_cnt_per_class.loc[x] = samples_cnt
#     last partition takes all the rest samples
    # print("partition {}: {} pct".format(str(len(partition_pcts)), partition_pcts[-1]))
    samples_cnt_last = [int(x-y) for x,y in zip(list(label_cnt_map.values()), df_samples_cnt_per_class.apply(lambda x: x.sum()))]
    df_samples_cnt_per_class.loc[len(partition_pcts)] = samples_cnt_last

    df_samples_cnt_per_class['total'] = df_samples_cnt_per_class.apply(lambda x: x.sum(),axis=1)
    # print(df_samples_cnt_per_class)

# Build a dataframe containing which indexes per class are in each node. Rows are partition_index, columns are class labels.
    df_indexes_per_class = pd.DataFrame(index=range(1,len(partition_pcts)+1), columns=labels.tolist())
    for label in labels:
        for i in list(df_samples_cnt_per_class.index.values) :
            part_len = df_samples_cnt_per_class.loc[i,label.item()]
            # print(part_len)
            # print(label_indexes_map[label.item()])
            # print(label_indexes_map[label.item()][:part_len])
            indexes_part = label_indexes_map[label.item()][:int(part_len)]
            # print(indexes_part)
            df_indexes_per_class.loc[i,label.item()] = indexes_part
            label_indexes_map[label.item()] = label_indexes_map[label.item()][int(part_len):]

    dataset_partition = GetPartitionedDataset(dataset, df_indexes_per_class, return_part_index)
    # show_samples_per_class(dataset_partition)

    return dataset_partition, df_samples_cnt_per_class, df_indexes_per_class
    # return df_samples_cnt_per_class, df_indexes_per_class