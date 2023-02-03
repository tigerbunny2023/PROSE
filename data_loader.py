import warnings
import pickle as pkl
import sys, os

import scipy.sparse as sp
import networkx as nx
import torch
import numpy as np

# from sklearn import datasets
# from sklearn.preprocessing import LabelBinarizer, scale
# from sklearn.model_selection import train_test_split
# from ogb.nodeproppred import DglNodePropPredDataset
# import copy

from utils import sparse_mx_to_torch_sparse_tensor #, dgl_graph_to_torch_sparse

warnings.simplefilter("ignore")


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)



def load_data(args):
    return load_citation_network(args.dataset, args.preprocess, args.sparse)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1)).astype(float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()



def new_load_data(data_path, args):
    print("Loading {} dataset...".format(args.dataset))

    features = sp.load_npz(data_path+args.dataset+"/feat.npz")

    if not args.preprocess:
        print("Does Not Preprocess Node Feature")
        features = features.todense()
    else:
        print("Proprecess Node Feature")
        features = preprocess_features(features)
    features = torch.FloatTensor(np.array(features))


    labels = torch.LongTensor(np.load(data_path+args.dataset+"/label.npy"))
    labels = torch.LongTensor(labels)


    idx_train = np.load(data_path+args.dataset+"/train.npy")
    idx_val = np.load(data_path+args.dataset+"/val.npy")
    idx_test = np.load(data_path+args.dataset+"/test.npy")
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    nfeats = features.shape[1]
    nclasses = torch.max(labels).item() + 1

    adj = sp.load_npz(data_path+args.dataset+"/ori_adj.npz")
    # adj_file_apth = data_path+args.dataset+"/wine_dele_0.8.npz"
    # adj = sp.load_npz(adj_file_apth)
    adj = np.array(adj.todense(),dtype='float32')

    return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj


def load_hori_pos(data_path, args):
    vec_pos = sp.load_npz(data_path+args.dataset+"/position.npz")
    return vec_pos
