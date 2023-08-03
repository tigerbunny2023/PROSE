import torch
import torch.nn as nn

EOS = 1e-10


class AnchorGCNLayer(nn.Module):
    """
    Simple AnchorGCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def sparse_mx_row_normalize(self, a_sparse):
        a_sparse = a_sparse.coalesce()
        inv_degree = 1.0 / torch.clamp(torch.sparse.sum(a_sparse, dim=-1).values(), min=VERY_SMALL_NUMBER)
        D_value = inv_degree[a_sparse.indices()[0]]
        new_values = a_sparse.values() * D_value
        anchor_norm = torch.sparse.FloatTensor(a_sparse.indices(), new_values, a_sparse.size())
        return anchor_norm

    def sparse_mx_col_normalize(self, a_sparse):
        a_sparse = a_sparse.coalesce()
        inv_degree = 1.0 / torch.clamp(torch.sparse.sum(a_sparse, dim=-2).values(), min=VERY_SMALL_NUMBER)
        D_value = inv_degree[a_sparse.indices()[1]]
        new_values = a_sparse.values() * D_value
        node_norm = torch.sparse.FloatTensor(a_sparse.indices(), new_values, a_sparse.size())
        return node_norm

    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(AnchorGCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, anchor_mp=False, batch_norm=False):
        support = torch.matmul(input, self.weight)

        if anchor_mp:
            node_anchor_adj = adj
            node_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-2, keepdim=True), min=1e-12) # N * anchor_num
            anchor_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-1, keepdim=True), min=1e-12) # N * anchor_num
            output = torch.matmul(anchor_norm, torch.matmul(node_norm.transpose(-1, -2), support))

            # node_anchor_adj = adj
            # node_norm = self.sparse_mx_col_normalize(node_anchor_adj)
            # anchor_norm = self.sparse_mx_row_normalize(node_anchor_adj)
            # output = torch.sparse.mm(anchor_norm, torch.sparse.mm(node_norm.transpose(-1, -2), support))
        else:
            node_adj = adj
            # output = torch.matmul(node_adj, support)
            output = torch.sparse.mm(node_adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())
