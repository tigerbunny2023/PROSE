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

from utils import sparse_mx_to_torch_sparse_tensor

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


def load_citation_network(dataset_str, preprocess=None ,sparse=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not sparse:
        adj = np.array(adj.todense(),dtype='float32')
    else:
        adj = sparse_mx_to_torch_sparse_tensor(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    if preprocess:
        print('--------preprocess node feature----------')
        features = preprocess_features(features)
        features = torch.FloatTensor(features)
    else:
        print('--------does not preprocess node feature----------')
        features = torch.FloatTensor(features.todense())

    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    nfeats = features.shape[1]
    for i in range(labels.shape[0]):
        sum_ = torch.sum(labels[i])
        if sum_ != 1:
            labels[i] = torch.tensor([1, 0, 0, 0, 0, 0])
    labels = (labels == 1).nonzero()[:, 1]
    nclasses = torch.max(labels).item() + 1

    return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj


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
    # adj_file_apth = data_path+args.dataset+"/cancer_dele_0.8.npz"
    # adj = sp.load_npz(adj_file_apth)
    adj = np.array(adj.todense(),dtype='float32')

    return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj


    # if args.ptb_feat == False:
    #     feature = sp.load_npz(data_path+dataset+"/feat.npz")
    # else:
    #     feature = sp.load_npz(ptb_path+"feat/"+dataset+"/"+"feat_"+str(args.ratio)+".npz")
    

    # if dataset == "cancer" or dataset == "digits" or dataset == "wine" or dataset == "wikics":
    #     feature = feature.todense()
    # else:
    #     feature = preprocess_features(feature)
    # feature = torch.FloatTensor(np.array(feature))

    # # feature = torch.FloatTensor(np.array(feature.todense()))
    
    # label = torch.LongTensor(np.load(data_path+dataset+"/label.npy"))
    # label = torch.LongTensor(label)
    # idx_train = np.load(data_path+dataset+"/train.npy")
    # idx_val = np.load(data_path+dataset+"/val.npy")
    # idx_test = np.load(data_path+dataset+"/test.npy")
    
    # ori_view = sp.load_npz(data_path+dataset+"/"+args.ori_view+".npz")
    # # ori_view1 = sp.load_npz(data_path+dataset+"/"+args.name_view1+".npz")
    # # ori_view2 = sp.load_npz(data_path+dataset+"/"+args.name_view2+".npz")
    # # ori_view1_indice = torch.load(data_path+dataset+"/"+args.indice_view1+".pt")
    # # ori_view2_indice = torch.load(data_path+dataset+"/"+args.indice_view2+".pt")

    # # if args.add:
    # #     if args.flag == 1:
    # #         ori_view1 = sp.load_npz(ptb_path+"add/"+dataset+"/"+str(args.ratio)+"_1.npz")
    # #     elif args.flag == 2:
    # #         ori_view2 = sp.load_npz(ptb_path+"add/"+dataset+"/"+str(args.ratio)+"_2.npz")
    # #     elif args.flag == 3:
    # #         ori_view1 = sp.load_npz(ptb_path+"add/"+dataset+"/"+str(args.ratio)+"_1.npz")
    # #         ori_view2 = sp.load_npz(ptb_path+"add/"+dataset+"/"+str(args.ratio)+"_2.npz")
    # # elif args.dele:
    # #     if args.flag == 1:
    # #         ori_view1 = sp.load_npz(ptb_path+"dele/"+dataset+"/"+str(args.ratio)+"_1.npz")
    # #     elif args.flag == 2:
    # #         ori_view2 = sp.load_npz(ptb_path+"dele/"+dataset+"/"+str(args.ratio)+"_2.npz")   
    # #     elif args.flag == 3:
    # #         ori_view1 = sp.load_npz(ptb_path+"dele/"+dataset+"/"+str(args.ratio)+"_1.npz")
    # #         ori_view2 = sp.load_npz(ptb_path+"dele/"+dataset+"/"+str(args.ratio)+"_2.npz") 

    # # ori_view1 = sparse_mx_to_torch_sparse_tensor(normalize_adj(ori_view1))
    # # ori_view2 = sparse_mx_to_torch_sparse_tensor(normalize_adj(ori_view2))

    # # position embedding
    # print("Loading {}'s position  info ...".format(dataset))
    # position_path = data_path+dataset+"/position.npz"
    # if not os.path.exists(position_path):
    #     distance_matrix = generation_position(position_path, ori_view, rat=1, max_distance= 8)
    # else:
    #     distance_matrix = sp.load_npz(position_path)
    
    # distance_matrix = sparse_mx_to_torch_sparse_tensor(distance_matrix)

    # ori_view = ori_view + sp.eye(feature.shape[0])
    # print(ori_view.todense())
    # ori_view = sparse_mx_to_torch_sparse_tensor(normalize_adj(ori_view))