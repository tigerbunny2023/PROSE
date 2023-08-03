import copy
import math

import torch

from graph_learners import *
from layers import AnchorGCNLayer
from torch.nn import Sequential, Linear, ReLU


class GCN_Sparse(nn.Module):
    def __init__(self, nfeat, nhid, nclass, graph_hops, dropout, batch_norm=False):
        super(GCN_Sparse, self).__init__()
        self.dropout = dropout

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(AnchorGCNLayer(nfeat, nhid, batch_norm=batch_norm))

        for _ in range(graph_hops - 2):
            self.graph_encoders.append(AnchorGCNLayer(nhid, nhid, batch_norm=batch_norm))

        self.graph_encoders.append(AnchorGCNLayer(nhid, nclass, batch_norm=False))


    def forward(self, x, node_anchor_adj):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = F.relu(encoder(x, node_anchor_adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, node_anchor_adj)

        return x



class Anchor_GraphEncoder(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, batch_norm = False, sparse = False):

        super(Anchor_GraphEncoder, self).__init__()
        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        # self.sparse = sparse

        self.gnn_encoder_layers = nn.ModuleList()
        self.gnn_encoder_layers.append(AnchorGCNLayer(in_dim, hidden_dim, batch_norm=batch_norm))
        for _ in range(nlayers - 2):
            self.gnn_encoder_layers.append(AnchorGCNLayer(hidden_dim, hidden_dim, batch_norm=batch_norm))
        
        self.gnn_encoder_layers.append(AnchorGCNLayer(hidden_dim, emb_dim, batch_norm=False))

        self.proj_head = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True),
                                           Linear(proj_dim, proj_dim))


    def forward(self, x, node_anchor_adj, anchor_mp=False, batch_norm=False):
        for i, encoder in enumerate(self.gnn_encoder_layers[:-1]):
            x = F.relu(encoder(x, node_anchor_adj, anchor_mp, batch_norm))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.gnn_encoder_layers[-1](x, node_anchor_adj, anchor_mp, batch_norm)
        z = self.proj_head(x)

        return z, x



class Anchor_GCL(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj):
        super(Anchor_GCL, self).__init__()
        
        self.encoder = Anchor_GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj)

    def forward(self, x, adj, anchor_mp=False, batch_norm=False):
        z, embedding = self.encoder(x, adj, anchor_mp, batch_norm)
        return z, embedding
    
    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2, sym=True):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:
            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
            return loss
        else:
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_1 = - torch.log(loss_1).mean()
            return loss_1
