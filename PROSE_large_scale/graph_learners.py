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


        self.weight_tensor = torch.Tensor(6, isize)
        self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))

        self.internal_layers = nn.ModuleList()
        print('Stage Internal type =', internal_type)
        self.internal_type = internal_type
        
        if self.internal_type == 'gnn':
            if nlayers == 1:
                self.internal_layers.append(AnchorGCNLayer(isize, osize))
            else:
                self.internal_layers.append(AnchorGCNLayer(isize, osize))
                for _ in range(nlayers - 2):
                    self.internal_layers.append(AnchorGCNLayer(osize, osize))
                self.internal_layers.append(AnchorGCNLayer(osize, osize))

        elif self.internal_type == 'mlp':
            if nlayers == 1:
                self.internal_layers.append(nn.Linear(isize, osize))
            else:
                self.internal_layers.append(nn.Linear(isize, osize))
                for _ in range(nlayers - 2):
                    self.internal_layers.append(nn.Linear(osize, osize))
                self.internal_layers.append(nn.Linear(osize, osize))

        # self.input_dim = isize
        # self.output_dim = osize
        self.k = k
        self.non_linearity = 'relu'
        self.i = i
        self.sparse = sparse
        self.act = act
        self.epsilon = epsilon
        self.fusion_ratio = fusion_ratio
        self.discrete_graph = discrete_graph
        self.add_embedding = add_embedding
        ## stage module
        self.ks = ks
        self.l_n = len(self.ks)


        if self.l_n > 0:
            self.stage_fusion_ratio = stage_fusion_ratio
            self.score_adj_normalize = score_adj_normalize
            # down score
            self.share_score = share_score
            if self.share_score:
                self.score_layer = AnchorGCNLayer(isize, 1)
            

            self.modify_subgraph = modify_subgraph

            ## up_gnn
            self.share_up_gnn = share_up_gnn
            self.gsl_adj_normalize = gsl_adj_normalize
            self.up_gnn_nlayers = up_gnn_nlayers
            if self.up_gnn_nlayers > 1:
                self.dropout_up_gnn = dropout_up_gnn
            
            # self.up_gnn_layers = nn.ModuleList()
            if self.share_up_gnn:
                self.up_gnn_layer = AnchorGCNLayer(isize, isize/2)
                # self.up_gnn_layers.append(AnchorGCNLayer(isize, osize))
                # if self.up_gnn_nlayers == 2:
                #     self.up_gnn_layers.append(AnchorGCNLayer(osize, osize))
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

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix


    def knn_anchor_node(self, context, anchors, k = 100, b = 500):
        expand_weight_tensor = self.weight_tensor.unsqueeze(1) # 6,1,32
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


    def forward_test(self, features, adj):
        embeddings = self.internal_forward(features, adj)
        embeddings_ = embeddings
        cross_layer_mi_val = None

        if self.l_n > 0:
            adj_ms = []
            indices_list = []
            down_outs = []

            for i in range(self.l_n): # [0,1,2]
                adj_ms.append(adj)
                down_outs.append(embeddings_)

                y = F.sigmoid(self.score_layer(embeddings, adj).squeeze())
                num_nodes = adj.shape[0]

                score, idx = torch.topk(y, max(2, int(self.ks[i]*num_nodes)))
                indices_list.append(idx)
                embeddings = embeddings[idx, :]
                # embeddings_ = torch.mul(embeddings, torch.unsqueeze(score, -1))
                embeddings_ = torch.mul(embeddings, torch.unsqueeze(score, -1) + torch.unsqueeze(1-score, -1).detach())

                adj = adj[idx, :][:, idx]

            if self.add_cross_mi:
                preds = []
                pre_g_vec = F.sigmoid(torch.mean(self.cross_mi_forward(embeddings_, adj),dim=0))
                for k in reversed(range(self.l_n)):
                    pos_g_vec = F.sigmoid(torch.mean(self.cross_mi_forward(down_outs[k], adj_ms[k]),dim=0))
                    neg_idx = np.random.permutation(down_outs[k].shape[0])
                    neg_g_vec = F.sigmoid(torch.mean(self.cross_mi_forward(down_outs[k][neg_idx], adj_ms[k]),dim=0))

                    pos_pred = self.discriminator(pos_g_vec, pre_g_vec)
                    neg_pred = self.discriminator(neg_g_vec, pre_g_vec)
                    preds.append(pos_pred)
                    preds.append(neg_pred)
                    pre_g_vec = pos_g_vec
                
                preds = torch.cat(preds,dim=0)
                labels = torch.tensor([1.0,0.0]).repeat(self.l_n).to(adj.device)
                cross_layer_mi_val = self.cross_layer_mi_loss(preds, labels)
                

        embeddings = F.normalize(embeddings_, dim=1, p=2)
        learned_adj = cal_similarity_graph(embeddings)
        learned_adj = top_k(learned_adj, self.k + 1)
        learned_adj = apply_non_linearity(learned_adj, self.non_linearity, self.i)


        if self.l_n > 0:
            for j in reversed(range(self.l_n)):
                learned_adj = symmetrize(learned_adj)
                learned_adj = normalize(learned_adj, 'sym', self.sparse)

                # fuse the origin graph and learn graph
                final_adj = self.fusion_ratio * learned_adj + (1-self.fusion_ratio) * adj

                # updata pre_layer subgraph based cur learned subgraph
                adj = self.stage_recover_adj(final_adj, adj_ms[j], indices_list[j])
                embeddings = down_outs[j]

                # excute graph neural netwrok based the small modified graph
                embeddings = self.up_gnn_layer(embeddings, adj)
                embeddings = F.normalize(embeddings, dim=1, p=2)
                learned_adj = cal_similarity_graph(embeddings)
                learned_adj = top_k(learned_adj, self.k + 1)
                learned_adj = apply_non_linearity(learned_adj, self.non_linearity, self.i)

        return learned_adj, cross_layer_mi_val



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
    

    def forward_sparse(self, features, adj):
        # Randomly sample s anchor nodes
        # init_node_vec = features
        # init_anchor_vec, sampled_node_idx = sample_anchors(init_node_vec, 700)
        # node_anchor_adj = self.knn_anchor_node(init_node_vec, init_anchor_vec, k=30, b=500)
        embeddings = self.internal_forward(features, adj)
        
        all_stage_adjs = []
        cross_layer_mi_val = None
        position_reg_loss = torch.tensor(0.0).to(features.device)

        rows, cols, values = knn_fast(embeddings, 30, 1000)
        rows_ = torch.cat((rows, cols))
        cols_ = torch.cat((cols, rows))
        values_ = torch.cat((values, values))
        values_ = apply_non_linearity(values_, self.non_linearity, self.i)
        
        indices = torch.cat((torch.unsqueeze(rows_, 0), torch.unsqueeze(cols_, 0)), 0)
        learned_adj = torch.sparse.FloatTensor(indices, values_)


        # learned_adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
        # learned_adj.edata['w'] = values_

        # fuse the origin graph and learn graph, and store
        # learned_adj_torch_sparse = dgl_graph_to_torch_sparse(learned_adj)
        # adj_torch_sparse = dgl_graph_to_torch_sparse(adj)

        prediction_adj = self.fusion_ratio * learned_adj + (1-self.fusion_ratio) * adj
        # prediction_adj = torch_sparse_to_dgl_graph(prediction_adj)
            
        all_stage_adjs.append(prediction_adj)

        return learned_adj, all_stage_adjs, cross_layer_mi_val, position_reg_loss



    def forward_anchor(self, features, init_anchor_vec, ori_adj):
        node_anchor_adj = self.knn_anchor_node(features, init_anchor_vec)
        node_anchor_adj = self.build_epsilon_neighbourhood(node_anchor_adj, 0.1, 0)

        if self.l_n > 0:
            adj_ms = []
            indices_list = []
            down_outs = []

            n_node = features.shape[0]
            pre_idx = torch.range(0, n_node-1).long()

            embeddings_ = features
            adj_ = ori_adj
            for i in range(self.l_n): # [0,1,2]
                # self.score_layer = nn.Linear(osize, 1)
                y = F.sigmoid(self.score_layer(embeddings_, adj_).squeeze())

                score, idx = torch.topk(y, max(2, int(self.ks[i]*adj_.shape[0])))
                _, indices = torch.sort(idx)
                new_score = score[indices]
                new_idx = idx[indices]
                # global node index
                pre_idx = pre_idx[new_idx]
                
                indices_list.append(pre_idx)

                adj_ = extract_subgraph(ori_adj, pre_idx)
                
                embeddings_ = embeddings_[new_idx, :]
                embeddings_ = torch.mul(embeddings_, torch.unsqueeze(new_score, -1) + torch.unsqueeze(1-new_score, -1).detach())

            node_anchor_vec = self.up_gnn_layer(features, node_anchor_adj, True, False)
            node_vec = self.up_gnn_layer(features, ori_adj, False, False)
            node_vec = torch.cat([node_anchor_vec, node_vec],dim=-1)
            
            new_node_anchor_adj = self.knn_anchor_node(node_vec, init_anchor_vec)
            new_node_anchor_adj = self.build_epsilon_neighbourhood(new_node_anchor_adj, 0.1, 0)

            for j in reversed(range(self.l_n)):    
                node_anchor_adj = self.only_modify_subgraph(new_node_anchor_adj, node_anchor_adj, indices_list[j], self.stage_fusion_ratio)

                node_anchor_vec = self.up_gnn_layer(features, node_anchor_adj, True, False)
                node_vec = self.up_gnn_layer(features, ori_adj, False, False)

                node_vec = torch.cat([node_anchor_vec, node_vec],dim=-1)

                new_node_anchor_adj = self.knn_anchor_node(node_vec, init_anchor_vec)
                node_anchor_adj = self.build_epsilon_neighbourhood(new_node_anchor_adj, 0.1, 0)
    
        return node_anchor_adj

    

    def forward(self, features, ori_adj):
        adj = ori_adj

        embeddings = self.internal_forward(features, adj)
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
                print(i)
                # adj_ms.append(adj_)
                down_outs.append(embeddings_)
                
                if self.share_score:
                    y = F.sigmoid(self.score_layer(embeddings_[pre_idx,:]).squeeze())

                score, idx = torch.topk(y, max(2, int(self.ks[i]*adj_.shape[0])))
                _, indices = torch.sort(idx)
                new_score = score[indices]
                new_idx = idx[indices]
                # global node index
                pre_idx = pre_idx[new_idx]
                
                indices_list.append(pre_idx)

                # for next subgraph selection
                # adj_ = adj[pre_idx, :][:, pre_idx]
                # adj_ = adj.index_select(0, pre_idx).index_select(1, pre_idx)
                
                # embeddings = embeddings[new_idx, :]
                # embeddings = torch.mul(embeddings, torch.unsqueeze(new_score, -1))
                # embeddings = torch.mul(embeddings, torch.unsqueeze(new_score, -1) + torch.unsqueeze(1-new_score, -1).detach())

                # for get the whole node embedding based rank scores
                mask_score = torch.zeros(n_node).to(features.device)
                mask_score[pre_idx] = new_score
                embeddings_ = torch.mul(embeddings_, torch.unsqueeze(mask_score, -1) + torch.unsqueeze(1-mask_score, -1).detach())

        rows, cols, values = knn_fast(embeddings_, self.k, 1000)
        rows_ = torch.cat((rows, cols))
        cols_ = torch.cat((cols, rows))
        values_ = torch.cat((values, values))
        values_ = apply_non_linearity(values_, self.non_linearity, self.i)
        indices = torch.cat((torch.unsqueeze(rows_, 0), torch.unsqueeze(cols_, 0)), 0)
        learned_adj = torch.sparse.FloatTensor(indices, values_)


        if self.l_n > 0:
            for j in reversed(range(self.l_n)):    
                if self.modify_subgraph:
                    adj = self.only_modify_subgraph(learned_adj, adj, indices_list[j], self.stage_fusion_ratio)
                else:
                    # only preserve the subgraph gradient
                    # learned_adj = self.mask_subgraph(indices_list[j], learned_adj)
                    # fuse the origin graph and learn graph
                    adj = self.fusion_ratio * learned_adj + (1-self.fusion_ratio) * adj

                # store learned graph adj
                all_stage_adjs.append(adj)

                # updata pre_layer subgraph based cur learned subgraph
                embeddings = down_outs[j]

                # 一层
                if self.gsl_adj_normalize:
                    embeddings = self.up_gnn_forward(features, normalize(adj, 'sym', self.sparse), deep=j)
                else:
                    embeddings = self.up_gnn_forward(features, adj, deep=j)
                
                rows, cols, values = knn_fast(embeddings, self.k, 1000)
                rows_ = torch.cat((rows, cols))
                cols_ = torch.cat((cols, rows))
                values_ = torch.cat((values, values))
                values_ = apply_non_linearity(values_, self.non_linearity, self.i)
                indices = torch.cat((torch.unsqueeze(rows_, 0), torch.unsqueeze(cols_, 0)), 0)
                learned_adj = torch.sparse.FloatTensor(indices, values_)

        prediction_adj = self.fusion_ratio * learned_adj + (1-self.fusion_ratio) * ori_adj
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
