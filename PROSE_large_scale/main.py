import argparse
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import load_data, new_load_data, load_ogb, load_coauthor
from model import GCN, GCL, GCN_Classifer, GCN_Sparse, Anchor_GCL
from graph_learners import *
from utils import *
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import dgl

import random

EOS = 1e-10

class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()


    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)
        # dgl.seed(seed)
        # dgl.random.seed(seed)


    def loss_cls(self, model, mask, features, adj, labels):
        logits = model(features, adj)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu

    def loss_cls_auc(self, model, mask, features, adj, labels):
        logits = model(features, adj)
        logp = F.log_softmax(logits, 1)
        
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        test_f1_macro, test_f1_micro, auc = self.gen_auc_mima(logp[mask], labels[mask])
        return loss, accu, test_f1_macro, test_f1_micro, auc

    
    def gen_auc_mima(self, logits, label):
        preds = torch.argmax(logits, dim=1)
        test_f1_macro = f1_score(label.cpu(), preds.cpu(), average='macro')
        test_f1_micro = f1_score(label.cpu(), preds.cpu(), average='micro')
        
        best_proba = F.softmax(logits, dim=1)
        if logits.shape[1] != 2:
            auc = roc_auc_score(y_true=label.detach().cpu().numpy(),
                                                    y_score=best_proba.detach().cpu().numpy(),
                                                    multi_class='ovr'
                                                    )
        else:
            auc = roc_auc_score(y_true=label.detach().cpu().numpy(),
                                                    y_score=best_proba[:,1].detach().cpu().numpy()
                                                    )
        return test_f1_macro, test_f1_micro, auc

    def loss_gcl(self, model, graph_learner, features, anchor_adj, mi_type = 'learn'):

        # view 1: anchor graph
        if args.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)

        z1, _ = model(features_v1, anchor_adj, 'anchor')

        # view 2: learned graph
        if args.maskfeat_rate_learner:
            mask, _ = get_feat_mask(features, args.maskfeat_rate_learner)
            features_v2 = features * (1 - mask)
        else:
            features_v2 = copy.deepcopy(features)

        learned_adj, all_stage_adjs, cross_mi_loss, position_reg_loss = graph_learner(features, anchor_adj)
        
        # if not args.sparse:
        #     learned_adj = symmetrize(learned_adj)
        #     learned_adj = normalize(learned_adj, 'sym', args.sparse)

        if mi_type == 'learn':
            z2, _ = model(features_v2, learned_adj, 'learner')
        elif mi_type == 'final':
            prediction_adj = all_stage_adjs[-1]
            z2, _ = model(features_v2, prediction_adj, 'learner')

        # compute loss
        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            # random.shuffle(node_idxs)
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss += model.calc_loss(z1[batch], z2[batch]) * weight
        else:
            loss = model.calc_loss(z1, z2)

        return loss, learned_adj, all_stage_adjs, cross_mi_loss, position_reg_loss


    def anchor_loss_gcl(self, model, ori_adj, features, node_anchor_adj):

        # view 1: anchor graph
        if args.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)

        z1, _ = model(features_v1, ori_adj, anchor_mp = False, batch_norm = False)

        # view 2: learned graph
        if args.maskfeat_rate_learner:
            mask, _ = get_feat_mask(features, args.maskfeat_rate_learner)
            features_v2 = features * (1 - mask)
        else:
            features_v2 = copy.deepcopy(features)


        z2, _ = model(features_v2, node_anchor_adj, anchor_mp = True, batch_norm = False)

        # compute loss
        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            # random.shuffle(node_idxs)
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss += model.calc_loss(z1[batch], z2[batch]) * weight
        else:
            loss = model.calc_loss(z1, z2)

        return loss


    def evaluate_adj_by_cls(self, Adj, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, args):

        model = GCN(in_channels=nfeats, hidden_channels=args.hidden_dim_cls, out_channels=nclasses, num_layers=args.nlayers_cls,
                    dropout=args.dropout_cls, dropout_adj=args.dropedge_cls, Adj=Adj, sparse=args.sparse)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)

        bad_counter = 0
        best_val = 0
        best_model = None

        if torch.cuda.is_available():
            model = model.cuda()
            train_mask = train_mask.cuda()
            val_mask = val_mask.cuda()
            test_mask = test_mask.cuda()
            features = features.cuda()
            labels = labels.cuda()

        for epoch in range(1, args.epochs_cls + 1):
            model.train()
            loss, accu = self.loss_cls(model, train_mask, features, labels)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if epoch % 10 == 0:
                model.eval()
                val_loss, accu = self.loss_cls(model, val_mask, features, labels)
                if accu > best_val:
                    bad_counter = 0
                    best_val = accu
                    best_model = copy.deepcopy(model)
                else:
                    bad_counter += 1

                if bad_counter >= args.patience_cls:
                    break
        best_model.eval()
        test_loss, test_accu = self.loss_cls(best_model, test_mask, features, labels)
        return best_val, test_accu, best_model

    
    def evaluate_adj_by_cls(self, model, mask, features, ori_adj, node_anchor_adj, labels):
        logits = model(features, adj)
        logp = F.log_softmax(logits, 1)
        
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        test_f1_macro, test_f1_micro, auc = self.gen_auc_mima(logp[mask], labels[mask])
        return loss, accu, test_f1_macro, test_f1_micro, auc
    
    def classifier_node_anchor(self, classifier, ori_adj, features, node_anchor_adj, fusion_ratio = 0.2):
        if fusion_ratio != 0.0:
            # Update node embeddings via node-anchor-node message passing
            init_agg_vec = classifier.graph_encoders[0](features, ori_adj, anchor_mp=False, batch_norm=False)
            node_vec = fusion_ratio * classifier.graph_encoders[0](features, node_anchor_adj, anchor_mp=True, batch_norm=False) + (1 - fusion_ratio) * init_agg_vec

            if classifier.graph_encoders[0].bn is not None:
                node_vec = classifier.graph_encoders[0].compute_bn(node_vec)

            node_vec = torch.relu(node_vec)
            node_vec = F.dropout(node_vec, classifier.dropout, training=classifier.training)
            

            # Add mid GNN layers
            for encoder in classifier.graph_encoders[1:-1]:
                node_vec = (1 - fusion_ratio) * encoder(node_vec, node_anchor_adj, anchor_mp=True, batch_norm=False) + \
                            fusion_ratio * encoder(node_vec, ori_adj, anchor_mp=False, batch_norm=False)

                if encoder.bn is not None:
                    node_vec = encoder.compute_bn(node_vec)

                node_vec = torch.relu(node_vec)
                node_vec = F.dropout(node_vec, classifier.dropout, training=classifier.training)

            # Compute output via node-anchor-node message passing
            output = fusion_ratio * classifier.graph_encoders[-1](node_vec, node_anchor_adj, anchor_mp=True, batch_norm=False) + (1 - fusion_ratio) * classifier.graph_encoders[-1](node_vec, ori_adj, anchor_mp=False, batch_norm=False)
        else:
            node_vec = classifier.graph_encoders[0](features, ori_adj, anchor_mp=False, batch_norm=False)
            if classifier.graph_encoders[0].bn is not None:
                node_vec = classifier.graph_encoders[0].compute_bn(node_vec)

            node_vec = torch.relu(node_vec)
            node_vec = F.dropout(node_vec, classifier.dropout, training=classifier.training)

            # Add mid GNN layers
            for encoder in classifier.graph_encoders[1:-1]:
                node_vec = encoder(node_vec, ori_adj, anchor_mp=False, batch_norm=False)

                if encoder.bn is not None:
                    node_vec = encoder.compute_bn(node_vec)

                node_vec = torch.relu(node_vec)
                node_vec = F.dropout(node_vec, classifier.dropout, training=classifier.training)

            output = classifier.graph_encoders[-1](node_vec, ori_adj, anchor_mp=False, batch_norm=False)

        return output

    def node_anchor_loss_cls_auc(self, classifier, ori_adj, features, node_anchor_adj, mask, labels, fusion_ratio = 0.3):
        logits = self.classifier_node_anchor(classifier, ori_adj, features, node_anchor_adj, fusion_ratio)
        logp = F.log_softmax(logits, 1)
        
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        test_f1_macro, test_f1_micro, auc = self.gen_auc_mima(logp[mask], labels[mask])
        return loss, accu, test_f1_macro, test_f1_micro, auc



    def train(self, args):
        print(args)

        torch.cuda.set_device(args.gpu)

        ## test load data
        data_path = "./dataset/"
        if args.dataset == "ogbn-arxiv":
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original = load_ogb(data_path, "ogbn-arxiv")
        elif args.dataset == 'Coauthor-CS' or args.dataset == 'Coauthor-Phy':
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original = load_coauthor(data_path, args.dataset)
        else:
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original = load_data(args)
        # features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original = new_load_data(data_path, args)

        test_accuracies = []
        validation_accuracies = []
        test_aucs = []
        validation_aucs = []
        test_f1_macros = []
        validation_f1_macros = []


        for trial in range(args.ntrials):
            print('seed = ', trial)
            self.setup_seed(trial)

            if args.sparse:
                anchor_adj_raw = adj_original
            else:
                anchor_adj_raw = torch.from_numpy(adj_original)

            anchor_adj = normalize(anchor_adj_raw + torch_sparse_eye(anchor_adj_raw.shape[0]), 'sym', args.sparse)
            # anchor_adj = normalize(anchor_adj_raw, 'sym', args.sparse)

            if args.sparse:
                anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)
                # anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)

            
            if torch.cuda.is_available():
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                test_mask = test_mask.cuda()
                features = features.cuda()
                labels = labels.cuda()
                # if not args.sparse:
                #     anchor_adj = anchor_adj.cuda()
                anchor_adj = anchor_adj.cuda()
            
            graph_learner = Stage_GNN_learner(1, features.shape[1], args.graph_learner_hidden_dim, args.k, args.sim_function, 6, args.sparse,
                                              args.activation_learner, args.internal_type, args.stage_ks, args.score_adj_normalize, args.share_score, args.share_up_gnn,
                                              args.fusion_ratio, args.stage_fusion_ratio, args.add_cross_mi, args.cross_mi_nlayer, args.epsilon, args.add_vertical_position,
                                              args.v_pos_dim, args.dropout_v_pos, args.position_regularization, args.discrete_graph, args.gsl_adj_normalize, 
                                              args.modify_subgraph, args.up_gnn_nlayers, args.dropout_up_gnn, args.add_embedding)

            model = Anchor_GCL(nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
                               emb_dim=args.rep_dim, proj_dim=args.proj_dim, dropout=args.dropout, 
                               dropout_adj=args.dropedge_rate)

            # classifier = GCN_Classifer(in_channels=nfeats, hidden_channels=args.hidden_dim_cls, out_channels=nclasses, num_layers=args.nlayers_cls,
            #                            dropout=args.dropout_cls, dropout_adj=args.dropedge_cls, sparse=args.sparse, batch_norm=args.bn_cls)
            
            # nfeat, nhid, nclass, graph_hops, dropout, batch_norm=False):
            classifier = GCN_Sparse(nfeat=nfeats, nhid=args.hidden_dim_cls, nclass=nclasses,
                                    graph_hops=args.nlayers_cls, dropout=args.dropout_cls, batch_norm=args.bn_cls)


            optimizer_cl = torch.optim.Adam(model.parameters(), lr=args.lr_cl, weight_decay=args.w_decay_cl)
            optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=args.lr_gsl, weight_decay=args.w_decay_gsl)
            optimizer_classifer = torch.optim.Adam(classifier.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)


            if torch.cuda.is_available():
                model = model.cuda()
                graph_learner = graph_learner.cuda()
                classifier = classifier.cuda()


            best_loss = 10000
            best_val = 0
            best_auc = 0
            best_macro_f1 = 0

            best_epch = 0
            bad_counter = 0
            best_adj = None
            best_classifier = None


            for epoch in range(1, args.epochs + 1):

                model.train()
                graph_learner.train()
                classifier.train()

                mi_loss = torch.zeros(1)

                # head & tail contrasitive loss
                # mi_loss, Adj, all_stage_adjs, cross_mi_loss, position_reg_loss = self.loss_gcl(model, graph_learner, features, anchor_adj, args.head_tail_mi_type)
                # learned_adj, all_stage_adjs, cross_mi_loss, position_reg_loss = graph_learner(features, anchor_adj)
                node_anchor_adj = None
                init_anchor_vec, sampled_node_idx = sample_anchors(features, 700)
                node_anchor_adj = graph_learner.forward_anchor(features, init_anchor_vec, anchor_adj)
                
                mi_loss = self.anchor_loss_gcl(model, anchor_adj, features, node_anchor_adj)
                # prediction_adj = all_stage_adjs[-1]
                # prediction_adj = normalize(prediction_adj, 'sym', args.sparse)
                # prediction_adj = anchor_adj
                # semi_loss, train_accu, train_f1_macro, train_f1_micro, train_auc = self.loss_cls_auc(classifier, train_mask, features, prediction_adj, labels)

                semi_loss, train_accu, train_f1_macro, train_f1_micro, train_auc = self.node_anchor_loss_cls_auc(classifier, anchor_adj, features, node_anchor_adj, train_mask, labels, 0.3)
                cur_anchor_adj = compute_anchor_adj(node_anchor_adj)
                semi_loss += add_graph_degree_loss(cur_anchor_adj)


                # if len(args.stage_ks)>0 and args.add_all_stage_loss:
                #     for i in all_stage_adjs[:-1]:
                #         stage_semi_loss, _, _, _, _ = self.loss_cls_auc(classifier, train_mask, features, prediction_adj, labels)
                #         semi_loss += stage_semi_loss
                #     semi_loss = semi_loss / (len(args.stage_ks) + 1)
                
                # semi_loss, accu = self.loss_cls(classifier, train_mask, features, Adj, labels)
                

                if args.head_tail_mi:
                    final_loss = semi_loss + mi_loss * args.mi_ratio
                else:
                    final_loss = semi_loss
                
                if args.add_cross_mi and (len(args.stage_ks)>0):
                    final_loss += cross_mi_loss * 0.005
                else:
                    cross_mi_loss = torch.zeros(1)

                if args.position_regularization:
                    final_loss += position_reg_loss * 0.005
                else:
                    position_reg_loss = torch.zeros(1)



                optimizer_cl.zero_grad()
                optimizer_learner.zero_grad()
                optimizer_classifer.zero_grad()

                # mi_loss.backward()
                # semi_loss.backward()
                final_loss.backward()

                optimizer_cl.step()
                optimizer_learner.step()
                optimizer_classifer.step()

                if epoch % args.eval_freq == 0:
                    classifier.eval()

                    val_loss, val_accu, val_f1_macro, val_f1_micro, val_auc = self.node_anchor_loss_cls_auc(classifier, anchor_adj, features, node_anchor_adj, val_mask, labels, 0.3)
                    test_loss, test_accu, test_f1_macro, test_f1_micro, test_auc = self.node_anchor_loss_cls_auc(classifier, anchor_adj, features, node_anchor_adj, test_mask, labels, 0.3)


                    # val_loss, val_accu, val_f1_macro, val_f1_micro, val_auc = self.loss_cls_auc(classifier, val_mask, features, prediction_adj, labels)
                    # test_loss, test_accu, test_f1_macro, test_f1_micro, test_auc = self.loss_cls_auc(classifier, test_mask, features, prediction_adj, labels)
                    
                    if best_loss >= val_loss and val_accu >= best_val:
                        print('--------------- update!----------------')
                        bad_counter = 0
                        best_epoch = epoch
                        best_loss = val_loss
                        best_val = val_accu
                        best_auc = val_auc
                        best_macro_f1 = val_f1_macro
                        # best_adj = prediction_adj
                        best_classifier = copy.deepcopy(classifier)
                    else:
                        bad_counter += 1

                    if bad_counter >= args.patience_cls:
                        break

                    print("Epoch {:05d} | Position_reg Loss {:.4f} | Cross_MI Loss {:.4f} |MI Loss {:.4f} | Eval Loss {:.4f} | Eval ACC {:.4f} | Test ACC {:.4f}| Eval Macro-F1 {:.4f} | Test Macro-F1 {:.4f}| Eval AUC  {:.4f} | Test AUC {:.4f}".format(
                        epoch, position_reg_loss.item(), cross_mi_loss.item(), mi_loss.item(), val_loss.item(), val_accu.item(), test_accu.item(), val_f1_macro.item(), test_f1_macro.item(), val_auc.item(), test_auc.item()))


                # Structure Bootstrapping
                # if (1 - args.tau) and (args.c == 0 or epoch % args.c == 0):
                #     n_n_adj = node_anchor_adj_2_node_node_adj(node_anchor_adj)
                #     anchor_adj = anchor_adj * args.tau + n_n_adj * (1 - args.tau)


                    # if args.sparse:
                    #     learned_adj_torch_sparse = dgl_graph_to_torch_sparse(Adj)
                    #     anchor_adj_torch_sparse = anchor_adj_torch_sparse * args.tau \
                    #                               + learned_adj_torch_sparse * (1 - args.tau)
                    #     anchor_adj = torch_sparse_to_dgl_graph(anchor_adj_torch_sparse)
                    # else:
                    #     anchor_adj = anchor_adj * args.tau + Adj.detach() * (1 - args.tau)
                    #     anchor_adj = normalize(anchor_adj, 'sym', args.sparse)

            
            best_classifier.eval()
            # test_loss, test_accu, test_f1_macro, test_f1_micro, test_auc = self.loss_cls_auc(best_classifier, test_mask, features, best_adj, labels)
            test_loss, test_accu, test_f1_macro, test_f1_micro, test_auc = self.node_anchor_loss_cls_auc(best_classifier, anchor_adj, features, node_anchor_adj, test_mask, labels, 0.3)
            
            print("Best Epoch {:05d} | Test Macro {:.4f} | Test Micro/ACC {:.4f} | AUC {:.4f}".format(best_epoch, test_f1_macro, test_f1_micro, test_auc))


            validation_accuracies.append(best_val.item())
            test_accuracies.append(test_accu.item())
                
            validation_aucs.append(best_auc.item())
            test_aucs.append(test_auc.item())
                
            validation_f1_macros.append(best_macro_f1.item())
            test_f1_macros.append(test_f1_macro.item())

            print("Trial: ", trial + 1)
            print("Best val ACC: ", best_val.item())
            print("Best test ACC: ", test_accu.item())
            print("Best val AUC: ", best_auc.item())
            print("Best test AUC: ", test_auc.item())
            print("Best val Macro F1: ", best_macro_f1.item())
            print("Best test Macro F1: ", test_f1_macro.item())


        if trial != 0:
            print('---------------------------results as follows------------------------------')
            self.print_results(validation_accuracies, test_accuracies, validation_aucs, test_aucs, validation_f1_macros, test_f1_macros)


    def print_results(self, validation_accu, test_accu, validation_aucs, test_aucs, validation_f1_macros, test_f1_macros):
        s_val = "Val accuracy: {:.4f} +/- {:.4f}".format(np.mean(validation_accu), np.std(validation_accu))
        s_test = "Test accuracy: {:.4f} +/- {:.4f}".format(np.mean(test_accu),np.std(test_accu))
        aucs_val = "Val auc: {:.4f} +/- {:.4f}".format(np.mean(validation_aucs), np.std(validation_aucs))
        aucs_test = "Test auc: {:.4f} +/- {:.4f}".format(np.mean(test_aucs),np.std(test_aucs))
        macro_f1_val = "Val macro f1: {:.4f} +/- {:.4f}".format(np.mean(validation_f1_macros),np.std(validation_f1_macros))
        macro_f1_test = "Test macro f1: {:.4f} +/- {:.4f}".format(np.mean(test_f1_macros),np.std(test_f1_macros))
        print(s_val)
        print(aucs_val)
        print(macro_f1_val)
        print(s_test)
        print(aucs_test)
        print(macro_f1_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Experimental setting
    parser.add_argument('-dataset', type=str, default='pubmed',
                        choices=['cora', 'citeseer', 'polblogs', 'wine', 'cancer', 'pubmed', 'ogbn-arxiv', 'Coauthor-CS', 'Coauthor-Phy'])
    parser.add_argument('-ntrials', type=int, default=5)
    parser.add_argument('-sparse', type=int, default=1)
    parser.add_argument('-eval_freq', type=int, default=1)
    parser.add_argument('-epochs', type=int, default=1000)
    parser.add_argument('-gpu', type=int, default=4)
    
    # New add =========================================
    parser.add_argument('-preprocess', type=int, default=1)
    parser.add_argument('-bn_cls', type=int, default=0)
    parser.add_argument('-add_embedding', type=int, default=0)
    parser.add_argument('-graph_learner_hidden_dim', type=int, default=32)
    parser.add_argument('-internal_type', type=str, default='gnn', choices=['gnn', 'mlp'])
    

    parser.add_argument('-epsilon', type=float, default=0.3)
    parser.add_argument('-fusion_ratio', type=float, default=0.1)
    parser.add_argument('-stage_fusion_ratio', type=float, default=0.05)


    parser.add_argument('-add_all_stage_loss', type=int, default=0)
    parser.add_argument('-stage_ks', nargs='+', type=list, default=[])
    parser.add_argument('-split_deep', type=int, default=0)
    parser.add_argument('-subgraph_ratio', type=float, default=0.7)

    
    parser.add_argument('-share_score', type=int, default=1)
    parser.add_argument('-score_adj_normalize', type=int, default=1)
    
    parser.add_argument('-share_up_gnn', type=int, default=1)
    parser.add_argument('-up_gnn_nlayers', type=int, default=2)
    parser.add_argument('-dropout_up_gnn', type=float, default=0.6)

    parser.add_argument('-modify_subgraph', type=int, default=0)

    parser.add_argument('-discrete_graph', type=int, default=0)
    parser.add_argument('-gsl_adj_normalize', type=int, default=0)

    parser.add_argument('-add_cross_mi', type=int, default=0)
    parser.add_argument('-cross_mi_nlayer', type=int, default=1)
    
    parser.add_argument('-head_tail_mi', type=int, default=1)
    parser.add_argument('-mi_ratio', type=float, default=0.1)
    parser.add_argument('-head_tail_mi_type', type=str, default='learn', choices=['learn', 'final'])

    parser.add_argument('-add_vertical_position', type=int, default=0)
    parser.add_argument('-v_pos_dim', type=int, default=16)
    parser.add_argument('-dropout_v_pos', type=float, default=0.5)

    parser.add_argument('-position_regularization', type=int, default=0)
    # New add end =========================================
    

    # GSL Module
    parser.add_argument('-lr_gsl', type=float, default=0.01)
    parser.add_argument('-w_decay_gsl', type=float, default=0.0005)
    parser.add_argument('-k', type=int, default=20)
    parser.add_argument('-sim_function', type=str, default='cosine', choices=['cosine', 'weight_cosine'])
    parser.add_argument('-activation_learner', type=str, default='relu', choices=["relu", "tanh"])

    # GCL Module -Augmentation
    parser.add_argument('-maskfeat_rate_learner', type=float, default=0.2)
    parser.add_argument('-maskfeat_rate_anchor', type=float, default=0.2)
    parser.add_argument('-dropedge_rate', type=float, default=0.25)


    # GCL Module - Framework
    parser.add_argument('-lr_cl', type=float, default=0.01)
    parser.add_argument('-w_decay_cl', type=float, default=0.0)
    parser.add_argument('-hidden_dim', type=int, default=64)
    parser.add_argument('-rep_dim', type=int, default=32)
    parser.add_argument('-proj_dim', type=int, default=16)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=1024)
    parser.add_argument('-nlayers', type=int, default=2)


    # Evaluation Network (Classification)
    parser.add_argument('-lr_cls', type=float, default=0.01)
    parser.add_argument('-w_decay_cls', type=float, default=0.0005)
    parser.add_argument('-hidden_dim_cls', type=int, default=64)
    parser.add_argument('-dropout_cls', type=float, default=0.5)
    parser.add_argument('-dropedge_cls', type=float, default=0.0)
    parser.add_argument('-nlayers_cls', type=int, default=2)
    parser.add_argument('-patience_cls', type=int, default=100)

    # Structure Bootstrapping
    parser.add_argument('-tau', type=float, default=0.9999)
    parser.add_argument('-c', type=int, default=0)

    args = parser.parse_args()

    if args.split_deep>0 and args.subgraph_ratio>0:
        args.stage_ks = [args.subgraph_ratio] * args.split_deep
        
    experiment = Experiment()
    experiment.train(args)
