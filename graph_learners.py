import dgl
import torch
import torch.nn as nn

from layers import Attentive, GCNConv_dense, GCNConv_dgl
from utils import *


class Stage_GNN_learner(nn.Module):
    def __init__(self, nlayers, isize, osize, k, knn_metric, i, sparse, act, internal_type, ks, score_adj_normalize, share_score, share_up_gnn, fusion_ratio, stage_fusion_ratio,
                 hori_pos, add_cross_mi, cross_mi_nlayer, epsilon, add_vertical_position, v_pos_dim, dropout_v_pos, position_regularization, 
                 discrete_graph, gsl_adj_normalize, modify_subgraph,up_gnn_nlayers, dropout_up_gnn, add_embedding):
        super(Stage_GNN_learner, self).__init__()

        self.internal_layers = nn.ModuleList()
        print('Stage Internal type =', internal_type)
        self.internal_type = internal_type
        if self.internal_type == 'gnn':
            if nlayers == 1:
                self.internal_layers.append(GCNConv_dense(isize, osize))
            else:
                self.internal_layers.append(GCNConv_dense(isize, osize))
                for _ in range(nlayers - 2):
                    self.internal_layers.append(GCNConv_dense(osize, osize))
                self.internal_layers.append(GCNConv_dense(osize, osize))
        elif self.internal_type == 'mlp':
            if nlayers == 1:
                self.internal_layers.append(nn.Linear(isize, osize))
            else:
                self.internal_layers.append(nn.Linear(isize, osize))
                for _ in range(nlayers - 2):
                    self.internal_layers.append(nn.Linear(osize, osize))
                self.internal_layers.append(nn.Linear(osize, osize))

        self.k = k
        self.non_linearity = 'relu'
        self.i = i
        self.sparse = sparse
        self.act = act
        self.epsilon = epsilon
        self.fusion_ratio = fusion_ratio
        self.discrete_graph = discrete_graph
        self.add_embedding = add_embedding
        self.hori_pos = hori_pos
        ## stage module
        self.ks = ks
        self.l_n = len(self.ks)


        if self.l_n > 0:
            if self.hori_pos:
                self.hori_pos_map = nn.Linear(self.hori_pos.shape[1], 16)
                self.hori_then_map = nn.Linear(osize+16, osize)

            self.stage_fusion_ratio = stage_fusion_ratio
            self.score_adj_normalize = score_adj_normalize
            # down score
            self.share_score = share_score
            if self.share_score:
                self.score_layer = GCNConv_dense(osize, 1)
            else:
                self.score_layers = nn.ModuleList()
                for _ in range(self.l_n):
                    self.score_layers.append(GCNConv_dense(osize, 1))

            self.modify_subgraph = modify_subgraph

            ## up_gnn
            self.share_up_gnn = share_up_gnn
            self.gsl_adj_normalize = gsl_adj_normalize
            self.up_gnn_nlayers = up_gnn_nlayers
            if self.up_gnn_nlayers > 1:
                self.dropout_up_gnn = dropout_up_gnn
            
            self.up_gnn_layers = nn.ModuleList()
            if self.share_up_gnn:
                self.up_gnn_layers.append(GCNConv_dense(osize, osize))
                if self.up_gnn_nlayers == 2:
                    self.up_gnn_layers.append(GCNConv_dense(osize, osize))
            else:
                for i in range(self.l_n):
                    self.up_gnn_layers_second = nn.ModuleList()
                    self.up_gnn_layers_second.append(GCNConv_dense(osize, osize))
                    if self.up_gnn_nlayers == 2:
                        self.up_gnn_layers_second.append(GCNConv_dense(osize, osize))
                    self.up_gnn_layers.append(self.up_gnn_layers_second)
            

            # cross layer mi
            self.add_cross_mi = add_cross_mi
            if self.add_cross_mi:

                self.discriminator = nn.Bilinear(osize, osize, 1)
                self.cross_layer_mi_loss = nn.BCEWithLogitsLoss()

                self.cross_mi_layers = nn.ModuleList()
                if cross_mi_nlayer == 1:
                    self.cross_mi_layers.append(GCNConv_dense(osize, osize))
                else:
                    self.cross_mi_layers.append(GCNConv_dense(osize, osize))
                    for _ in range(cross_mi_nlayer - 2):
                        self.cross_mi_layers.append(
                            GCNConv_dense(osize, osize))
                    self.cross_mi_layers.append(GCNConv_dense(osize, osize))


            # vectival position
            self.add_vertical_position = add_vertical_position
            if self.add_vertical_position:
                self.dropout_v_pos = dropout_v_pos
                self.v_pos_dim = v_pos_dim
                self.vertival_pos_embedding = nn.Embedding(
                    self.l_n+1, self.v_pos_dim)
                self.map_v_pos_linear1 = nn.Linear(osize+self.v_pos_dim, osize)
                self.map_v_pos_linear2 = nn.Linear(osize, osize)

            self.position_regularization = position_regularization


    def up_gnn_forward(self, h, adj, deep=None):
        if self.share_up_gnn:
            for i, up_layer in enumerate(self.up_gnn_layers):
                h = up_layer(h, adj)
                if i != (len(self.up_gnn_layers) - 1):
                    h = F.relu(h)
                    if self.dropout_up_gnn>0:
                        h = F.dropout(h, self.dropout_up_gnn, training=self.training)
            return h
        else:
            up_gnn_layers = self.up_gnn_layers[deep]
            for i, up_layer in enumerate(up_gnn_layers):
                h = up_layer(h, adj)
                if i != (len(up_gnn_layers) - 1):
                    h = F.relu(h)
                    if self.dropout_up_gnn>0:
                        h = F.dropout(h, self.dropout_up_gnn, training=self.training)
            return h



    def internal_forward(self, h, adj):
        if self.internal_type == 'gnn':
            for i, layer in enumerate(self.internal_layers):
                h = layer(h, adj)
                if i != (len(self.internal_layers) - 1):
                    if self.act == "relu":
                        h = F.relu(h)
                    elif self.act == "tanh":
                        h = F.tanh(h)
            return h
        elif self.internal_type == 'mlp':
            for i, layer in enumerate(self.internal_layers):
                h = layer(h)
                if i != (len(self.internal_layers) - 1):
                    if self.act == "relu":
                        h = F.relu(h)
                    elif self.act == "tanh":
                        h = F.tanh(h)
            return h
    
    
    def cross_mi_forward(self, h, adj):
        assert (self.l_n>0 and self.add_cross_mi)

        for i, layer in enumerate(self.cross_mi_layers):
            h = layer(h, adj)
            if i != (len(self.cross_mi_layers) - 1):
                if self.act == "relu":
                    h = F.relu(h)
                elif self.act == "tanh":
                    h = F.tanh(h)
        return h



    def features_gsl(self, embeddings):
        embeddings_ = F.normalize(embeddings, dim=1, p=2)
        learned_adj = cal_similarity_graph(embeddings)
        learned_adj = top_k(learned_adj, self.k + 1)
        mask = (learned_adj > self.epsilon).float()
        mask.requires_grad = False
        learned_adj = learned_adj * mask
        return learned_adj

    def to_discrete_graph(self, adj_score):
        mask = (adj_score > 0).float()
        new_adj = adj_score + (1-adj_score).detach() * mask
        return new_adj
    
    def forward(self, features, adj, hori_pos=None):
        pos_infos = (adj>0).float()

        embeddings = self.internal_forward(features, adj)
        if hori_pos:
            hori_pos_ = self.hori_pos_map(hori_pos)
            cur_embeddings = torch.cat((embeddings, hori_pos_), dim=1)
            cur_embeddings = self.hori_then_map(cur_embeddings)
        else:
            cur_embeddings = embeddings

        adj_ = adj
        embeddings_ = embeddings

        all_stage_adjs = []


        cross_layer_mi_val = None
        position_reg_loss = torch.tensor(0.0).to(features.device)

        if self.l_n > 0:
            adj_ms = []
            indices_list = []
            down_outs = []

            n_node = features.shape[0]
            pre_idx = torch.range(0, n_node-1).long()
            for i in range(self.l_n): # [0,1,2]
                adj_ms.append(adj_)
                down_outs.append(embeddings_)
                
                if self.share_score:
                    if i == 0:
                        y = F.sigmoid(self.score_layer(embeddings_, adj_).squeeze())
                    else:
                        if self.score_adj_normalize:
                            y = F.sigmoid(self.score_layer(embeddings_[pre_idx,:], normalize(adj_, 'sym', self.sparse)).squeeze())
                        else:
                            y = F.sigmoid(self.score_layer(embeddings_[pre_idx,:], adj_).squeeze())
                else:
                    if self.score_adj_normalize:
                        y = F.sigmoid(self.score_layers[i](embeddings_[pre_idx,:], normalize(adj_, 'sym', self.sparse)).squeeze())
                    else:
                        y = F.sigmoid(self.score_layers[i](embeddings_[pre_idx,:], adj_).squeeze())
                
                score, idx = torch.topk(y, max(2, int(self.ks[i]*adj_.shape[0])))

                _, indices = torch.sort(idx)
                new_score = score[indices]
                new_idx = idx[indices]
                
                # global node index
                pre_idx = pre_idx[new_idx]

                if self.position_regularization:
                    position_reg_loss += self.anchor_position_regularization(pos_infos[pre_idx, :], new_score)
                
                indices_list.append(pre_idx)

                # for next subgraph selection
                adj_ = adj[pre_idx, :][:, pre_idx]

                # embeddings = embeddings[new_idx, :]
                # embeddings = torch.mul(embeddings, torch.unsqueeze(new_score, -1))
                # embeddings = torch.mul(embeddings, torch.unsqueeze(new_score, -1) + torch.unsqueeze(1-new_score, -1).detach())

                # for get the whole node embedding based rank scores
                mask_score = torch.zeros(n_node).to(features.device)
                mask_score[pre_idx] = new_score
                embeddings_ = torch.mul(embeddings_, torch.unsqueeze(mask_score, -1) + torch.unsqueeze(1-mask_score, -1).detach())
                

            if self.add_cross_mi:
                preds = []
                pre_g_vec = F.sigmoid(torch.mean(self.cross_mi_forward(embeddings_[pre_idx,:], normalize(adj_, 'sym', self.sparse)),dim=0))
                for k in reversed(range(self.l_n)):
                    if k==0:
                        pos_g_vec = F.sigmoid(torch.mean(self.cross_mi_forward(down_outs[k], adj_ms[k]),dim=0))
                        neg_idx = np.random.permutation(down_outs[k].shape[0])
                        neg_g_vec = F.sigmoid(torch.mean(self.cross_mi_forward(down_outs[k][neg_idx], adj_ms[k]),dim=0))
                    else:
                        pos_g_vec = F.sigmoid(torch.mean(self.cross_mi_forward(down_outs[k][indices_list[k-1],:], normalize(adj_ms[k], 'sym', self.sparse)),dim=0))
                        neg_idx = np.random.permutation(indices_list[k-1].shape[0])
                        neg_g_vec = F.sigmoid(torch.mean(self.cross_mi_forward(down_outs[k][indices_list[k-1],:][neg_idx], normalize(adj_ms[k], 'sym', self.sparse)),dim=0))

                    pos_pred = self.discriminator(pos_g_vec, pre_g_vec)
                    neg_pred = self.discriminator(neg_g_vec, pre_g_vec)
                    preds.append(pos_pred)
                    preds.append(neg_pred)
                    pre_g_vec = pos_g_vec
                
                preds = torch.cat(preds,dim=0)
                labels = torch.tensor([1.0,0.0]).repeat(self.l_n).to(adj.device)
                cross_layer_mi_val = self.cross_layer_mi_loss(preds, labels)
                

            if self.add_vertical_position:
                vertical_position = torch.zeros(n_node).long().to(adj.device)
                for i in range(self.l_n):
                    vertical_position[indices_list[i]] = int(i+1)
                
                node_v_pos_embeddings = self.vertival_pos_embedding(vertical_position)
                
                embeddings_ = torch.cat((embeddings_, node_v_pos_embeddings), dim=-1)
                embeddings_ = F.relu(self.map_v_pos_linear1(embeddings_))
                embeddings_ = F.dropout(embeddings_, self.dropout_v_pos, training=self.training)
                embeddings_ = self.map_v_pos_linear2(embeddings_)


        if self.add_embedding:
            embeddings_ += cur_embeddings
        embeddings = F.normalize(embeddings_, dim=1, p=2)
        learned_adj = cal_similarity_graph(embeddings)
        if self.k:
            learned_adj = top_k(learned_adj, self.k + 1)
        mask = (learned_adj > self.epsilon).float()
        mask.requires_grad = False
        learned_adj = learned_adj * mask


        if self.l_n > 0:
            for j in reversed(range(self.l_n)):
                learned_adj = symmetrize(learned_adj)
                if self.discrete_graph:
                    learned_adj = self.to_discrete_graph(learned_adj)
                learned_adj = normalize(learned_adj, 'sym', self.sparse)
                
                if self.modify_subgraph:
                    adj = self.only_modify_subgraph(learned_adj, adj, indices_list[j], self.stage_fusion_ratio)
                else:
                    # only preserve the subgraph gradient
                    learned_adj = self.mask_subgraph(indices_list[j], learned_adj)
                    # fuse the origin graph and learn graph
                    adj = self.fusion_ratio * learned_adj + (1-self.fusion_ratio) * adj

                # store learned graph adj
                all_stage_adjs.append(adj)

                # updata pre_layer subgraph based cur learned subgraph
                embeddings = down_outs[j]
                if self.add_vertical_position:
                    embeddings = torch.cat((embeddings, node_v_pos_embeddings), dim=-1)
                    embeddings = F.relu(self.map_v_pos_linear1(embeddings))
                    embeddings = F.dropout(embeddings_, self.dropout_v_pos, training=self.training)
                    embeddings = self.map_v_pos_linear2(embeddings)

                if self.gsl_adj_normalize:
                    embeddings = self.up_gnn_forward(embeddings, normalize(adj, 'sym', self.sparse), deep=j)
                else:
                    embeddings = self.up_gnn_forward(embeddings, adj, deep=j)

                if self.add_embedding:
                    embeddings += cur_embeddings
                embeddings = F.normalize(embeddings, dim=1, p=2)
                learned_adj = cal_similarity_graph(embeddings)
                if self.k:
                    learned_adj = top_k(learned_adj, self.k + 1)

                # filter the elements below epsilon
                mask = (learned_adj > self.epsilon).float()
                mask.requires_grad = False
                learned_adj = learned_adj * mask
            
        learned_adj = symmetrize(learned_adj)
        if self.discrete_graph:
            learned_adj = self.to_discrete_graph(learned_adj)
        learned_adj = normalize(learned_adj, 'sym', self.sparse)

        # fuse the origin graph and learn graph, and store
        prediction_adj = self.fusion_ratio * learned_adj + (1-self.fusion_ratio) * adj
        # prediction_adj = normalize(prediction_adj, 'sym', self.sparse)
            
        all_stage_adjs.append(prediction_adj)

        return learned_adj, all_stage_adjs, cross_layer_mi_val, position_reg_loss


    def anchor_position_regularization(self, anchor_postions, socres):
        node_num = anchor_postions.shape[0]
        mask = torch.eye(node_num).to(anchor_postions.device)
        pos_scores = torch.mul(anchor_postions, torch.unsqueeze(socres, -1))
        scores_norm = pos_scores.div(torch.norm(pos_scores, p=2, dim=-1, keepdim=True)+1e-12)
        pos_loss = torch.mm(scores_norm, scores_norm.transpose(-1, -2)) * (1-mask)
        return torch.sum(pos_loss) / node_num


    def stage_recover_adj(self, cur_small_g, pre_big_g, idx):
            n_nums = idx.shape[0]
            # x_index = idx.unsqueeze(1).repeat(1,n_nums).flatten()
            # y_index = idx.repeat(1,n_nums).flatten()
            x_index = idx.repeat(n_nums)
            y_index = idx.repeat_interleave(n_nums)
            cur_adj_v = cur_small_g.flatten()
            new_pre_adj = pre_big_g.index_put([x_index,y_index],cur_adj_v)
            return new_pre_adj


    def mask_subgraph(self, idx, adj):
        n_node = adj.shape[0]
        n_nums = idx.shape[0]
        x_index = idx.repeat(n_nums)
        y_index = idx.repeat_interleave(n_nums)

        mask = torch.zeros(n_node, n_node).to(adj.device)
        mask[x_index, y_index] = 1.0

        mask_adj = adj * mask + ((1-mask) * adj).detach()
        return mask_adj

    def only_modify_subgraph(self, cur_g, pre_g, idx, fusion_ratio):
        cur_small_g = cur_g[idx,:][:,idx]
        pre_small_g = pre_g[idx,:][:,idx]
        new_small_g = cur_small_g * fusion_ratio + pre_small_g * (1-fusion_ratio)

        new_g = self.stage_recover_adj(new_small_g, pre_g, idx)
        return new_g
