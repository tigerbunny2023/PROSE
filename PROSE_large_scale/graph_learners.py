import dgl
import torch
import torch.nn as nn

from layers import Attentive, GCNConv_dense, GCNConv_dgl, AnchorGCNLayer
from utils import *



class Stage_GNN_learner(nn.Module):
    def __init__(self, nlayers, isize, osize, k, knn_metric, i, sparse, act, internal_type, ks, score_adj_normalize, share_score, share_up_gnn, fusion_ratio, stage_fusion_ratio,
                 add_cross_mi, cross_mi_nlayer, epsilon, add_vertical_position, v_pos_dim, dropout_v_pos, position_regularization, 
                 discrete_graph, gsl_adj_normalize, modify_subgraph,up_gnn_nlayers, dropout_up_gnn, add_embedding):
        super(Stage_GNN_learner, self).__init__()


        self.weight_tensor1 = torch.Tensor(6, isize)
        self.weight_tensor1 = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor1))

        self.weight_tensor2 = torch.Tensor(6, osize)
        self.weight_tensor2 = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor2))

        self.sparse = sparse
        self.act = act
        self.epsilon = epsilon
        self.fusion_ratio = fusion_ratio
        ## stage module
        self.ks = ks
        self.l_n = len(self.ks)

        if self.l_n > 0:
            # down score
            self.score_layer = AnchorGCNLayer(isize, 1)

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix


    def knn_anchor_node(self, context, anchors, weight_tensor, k = 100, b = 500):
        expand_weight_tensor = weight_tensor.unsqueeze(1) # 6,1,32
        if len(context.shape) == 3:
            expand_weight_tensor = expand_weight_tensor.unsqueeze(1)
            
        # context -  N * 32
        # context.unsqueeze(0) - 1 * N * 32
        context_fc = context.unsqueeze(0) * expand_weight_tensor
        # context_fc - 6 * N * 32
        context_norm = F.normalize(context_fc, p=2, dim=-1)
        # context_norm - 6 * N * 32

        anchors_fc = anchors.unsqueeze(0) * expand_weight_tensor
        anchors_norm = F.normalize(anchors_fc, p=2, dim=-1)  # 6 * anchor_num * 32

        attention = torch.matmul(context_norm, anchors_norm.transpose(-1, -2)).mean(0)
        markoff_value = 0

        # index = 0
        # values = torch.zeros(context_norm.shape[1] * (k + 1)).cuda()
        # rows = torch.zeros(context_norm.shape[1] * (k + 1)).cuda()
        # cols = torch.zeros(context_norm.shape[1] * (k + 1)).cuda()
        # # norm_row = torch.zeros(context_norm.shape[1]).cuda()
        # # norm_col = torch.zeros(context_norm.shape[1]).cuda()
        # while index < context_norm.shape[1]:
        #     if (index + b) > (context_norm.shape[1]):
        #         end = context_norm.shape[1]
        #     else:
        #         end = index + b
        #     sub_tensor = context_norm[:,index:index + b,:]
        #     # similarities = torch.matmul(sub_tensor, context_norm.transpose(-1, -2)).mean(0)
        #     similarities = torch.matmul(sub_tensor, anchors_norm.transpose(-1, -2)).mean(0)
        #     #------start---------
        #     similarities_ = self.build_epsilon_neighbourhood(similarities, 0.1, markoff_value)
        #     # or inds
        #     # #-------end--------
        #     vals, inds = similarities_.topk(k=k + 1, dim=-1)
        #     values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        #     cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        #     rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        #     # norm_row[index: end] = torch.sum(vals, dim=1)
        #     # norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        #     index += b
        # rows = rows.long()
        # cols = cols.long()

        # # rows_ = torch.cat((rows, cols))
        # # cols_ = torch.cat((cols, rows))
        # # values_ = torch.cat((values, values))
        # values_ = F.relu(values)
        # indices = torch.cat((torch.unsqueeze(rows, 0), torch.unsqueeze(cols, 0)), 0)
        # attention = torch.sparse.FloatTensor(indices, values_)
        return attention


    def forward_anchor(self, features, ori_adj, anchor_nodes_idx, encoder, fusion_ratio):
        node_anchor_adj = self.knn_anchor_node(features, features[anchor_nodes_idx], self.weight_tensor1)
        node_anchor_adj = self.build_epsilon_neighbourhood(node_anchor_adj, 0.1, 0)

        if self.l_n > 0:
            indices_list = []

            n_node = features.shape[0]
            pre_idx = torch.range(0, n_node-1).long()

            embeddings_ = features
            adj_ = ori_adj
            for i in range(self.l_n): # [0,1,2]
                # self.score_layer = nn.Linear(osize, 1)
                y = F.sigmoid(self.score_layer(embeddings_[pre_idx,:], adj_).squeeze())

                score, idx = torch.topk(y, max(2, int(self.ks[i]*adj_.shape[0])))
                _, indices = torch.sort(idx)
                new_score = score[indices]
                new_idx = idx[indices]

                # global node index
                pre_idx = pre_idx[new_idx]
                
                indices_list.append(pre_idx)

                adj_ = extract_subgraph(adj_, new_idx)
                

                mask_score = torch.zeros(n_node).to(features.device)
                mask_score[pre_idx] = new_score
                embeddings_ = torch.mul(embeddings_, torch.unsqueeze(mask_score, -1) + torch.unsqueeze(1-mask_score, -1).detach())


            for j in reversed(range(self.l_n)):
                node_anchor_vec = encoder(embeddings_, node_anchor_adj, True, False)
                node_vec = encoder(embeddings_, ori_adj, False, False)
                node_vec = fusion_ratio * node_anchor_vec + (1 - fusion_ratio) * node_vec
                
                new_node_anchor_adj = self.knn_anchor_node(node_vec, node_vec[anchor_nodes_idx], self.weight_tensor2)
                new_node_anchor_adj = self.build_epsilon_neighbourhood(new_node_anchor_adj, 0.1, 0)

                # modify the node_anchor subgraph
                mask = torch.ones(n_node).to(features.device)
                mask[indices_list[j]] = 0.95
                node_anchor_adj = torch.mul(node_anchor_adj, torch.unsqueeze(mask, -1)) + torch.mul(new_node_anchor_adj, torch.unsqueeze(1-mask, -1).detach())
            
        return node_anchor_adj
