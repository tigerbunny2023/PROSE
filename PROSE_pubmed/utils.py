import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
# from sklearn import metrics

EOS = 1e-10

def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'none':
        return tensor
    else:
        raise NameError('We dont support the non-linearity yet')


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def edge_deletion(adj, drop_r):
    edge_index = np.array(np.nonzero(adj))
    half_edge_index = edge_index[:, edge_index[0,:] < edge_index[1,:]]
    num_edge = half_edge_index.shape[1]
    samples = np.random.choice(num_edge, size=int(drop_r * num_edge), replace=False)
    dropped_edge_index = half_edge_index[:, samples].T
    adj[dropped_edge_index[:,0],dropped_edge_index[:,1]] = 0.
    adj[dropped_edge_index[:,1],dropped_edge_index[:,0]] = 0.
    return adj

def edge_addition(adj, add_r):
    edge_index = np.array(np.nonzero(adj))
    half_edge_index = edge_index[:, edge_index[0,:] < edge_index[1,:]]
    num_edge = half_edge_index.shape[1]
    num_node = adj.shape[0]
    added_edge_index_in = np.random.choice(num_node, size=int(add_r * num_edge), replace=True)
    added_edge_index_out = np.random.choice(num_node, size=int(add_r * num_edge), replace=True)
    adj[added_edge_index_in,added_edge_index_out] = 1.
    adj[added_edge_index_out,added_edge_index_in] = 1.
    return adj


def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    return mask.cuda(), samples


def accuracy(preds, labels):
    pred_class = torch.max(preds, 1)[1]
    return torch.sum(torch.eq(pred_class, labels)).float() / labels.shape[0]


def normalize(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            aa = torch.sparse.sum(adj, dim=1)
            bb = aa.values()
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


def symmetrize(adj):  # only for non-sparse
    return (adj + adj.T) / 2

def sample_anchors(node_vec, s):
    idx = torch.randperm(node_vec.size(0))[:s]
    return node_vec[idx], idx


def add_graph_degree_loss(out_adj, weight=0.03):
    # Graph regularization
    graph_loss = 0
    # L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj
    # graph_loss += self.config['smoothness_ratio'] * torch.trace(torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
    ones_vec = torch.ones(out_adj.size(-1)).cuda()
    graph_loss = - weight * torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(out_adj, ones_vec.unsqueeze(-1)) + 1e-12)).squeeze() / out_adj.shape[-1]
    return graph_loss

def node_anchor_adj_2_node_node_adj(node_anchor_adj):
    node_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-2, keepdim=True), min=1e-12) # N * anchor_num
    anchor_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-1, keepdim=True), min=1e-12) # N * anchor_num
    output = torch.matmul(anchor_norm.detach().cpu(), node_norm.transpose(-1, -2).detach().cpu())
    return output.cuda()

def compute_anchor_adj(node_anchor_adj, anchor_mask=None):
    '''Can be more memory-efficient'''
    anchor_node_adj = node_anchor_adj.transpose(-1, -2)
    anchor_norm = torch.clamp(anchor_node_adj.sum(dim=-2), min=1e-12) ** -1
    # anchor_adj = torch.matmul(anchor_node_adj, torch.matmul(torch.diag(anchor_norm), node_anchor_adj))
    anchor_adj = torch.matmul(anchor_node_adj, anchor_norm.unsqueeze(-1) * node_anchor_adj)

    markoff_value = 0
    if anchor_mask is not None:
        anchor_adj = anchor_adj.masked_fill_(1 - anchor_mask.byte().unsqueeze(-1), markoff_value)
        anchor_adj = anchor_adj.masked_fill_(1 - anchor_mask.byte().unsqueeze(-2), markoff_value)

    return anchor_adj


def cal_similarity_graph(node_embeddings):
    similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    return similarity_graph


def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape).cuda()
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph


def knn_fast(X, k, b):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).cuda()
    rows = torch.zeros(X.shape[0] * (k + 1)).cuda()
    cols = torch.zeros(X.shape[0] * (k + 1)).cuda()
    norm_row = torch.zeros(X.shape[0]).cuda()
    norm_col = torch.zeros(X.shape[0]).cuda()
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def extract_subgraph(sparse_tensor, node_indices):
    node_mask = torch.zeros(sparse_tensor.size(0), dtype=torch.bool)
    node_mask[node_indices] = True
    
    edge_indices = []
    edge_values = []

    index_map = torch.zeros(sparse_tensor.size(0), dtype=torch.long)
    index_map[node_indices] = torch.arange(node_indices.size(0), dtype=torch.long)

    for i, j in zip(sparse_tensor._indices().t().tolist(), sparse_tensor._values().tolist()):
        if node_mask[i[0]] and node_mask[i[1]]:
            edge_indices.append([index_map[i[k]] for k in range(2)])
            edge_values.append(j)
    
    if len(edge_indices) > 0:
        sub_sparse = torch.sparse_coo_tensor(
            torch.tensor(edge_indices).t(), torch.tensor(edge_values), 
            (len(node_indices), len(node_indices)),
            dtype=sparse_tensor.dtype, device=sparse_tensor.device)
    else:
        sub_sparse = None
        
    return sub_sparse


def torch_sparse_eye(num_nodes):
    indices = torch.arange(num_nodes).repeat(2, 1)
    values = torch.ones(num_nodes)
    return torch.sparse.FloatTensor(indices, values)
