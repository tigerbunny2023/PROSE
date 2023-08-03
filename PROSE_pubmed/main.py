import argparse
import copy

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import load_data, load_ogb, load_coauthor
from model import GCN_Sparse, Anchor_GCL
from graph_learners import *
from utils import *
from sklearn.metrics import f1_score, roc_auc_score

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

        ## load data
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original = load_data(args)
        # features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original = new_load_data(data_path, args)

        test_accuracies = []
        validation_accuracies = []
        test_aucs = []
        validation_aucs = []
        test_f1_macros = []
        validation_f1_macros = []


        for trial in range(args.ntrials):
            print('seed = ', args.seeds[trial])
            self.setup_seed(args.seeds[trial])


            if args.sparse:
                anchor_adj_raw = adj_original
            else:
                anchor_adj_raw = torch.from_numpy(adj_original)

            anchor_adj = normalize(anchor_adj_raw + torch_sparse_eye(anchor_adj_raw.shape[0]), 'sym', args.sparse)

            if args.sparse:
                anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)

            if torch.cuda.is_available():
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                test_mask = test_mask.cuda()
                features = features.cuda()
                labels = labels.cuda()
                anchor_adj = anchor_adj.cuda()
            
            
            graph_learner = Stage_GNN_learner(features.shape[1], args.hidden_dim_cls, args.head_num, args.sparse,
                                              args.stage_ks, args.anchor_adj_fusion_ratio, args.epsilon)

            model = Anchor_GCL(nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
                               emb_dim=args.rep_dim, proj_dim=args.proj_dim, dropout=args.dropout, 
                               dropout_adj=args.dropedge_rate)

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
            best_node_anchor_adj = None



            for epoch in range(1, args.epochs + 1):

                model.train()
                graph_learner.train()
                classifier.train()

                mi_loss = torch.zeros(1)

                node_anchor_adj = None
                # init_anchor_vec, sampled_node_idx = sample_anchors(features, 700)
                anchor_nodes_idx = torch.randperm(features.size(0))[:args.anchor_num]
                node_anchor_adj = graph_learner.forward_anchor(features, anchor_adj, anchor_nodes_idx, classifier.graph_encoders[0], args.emb_fusion_ratio)
                
                # mi_loss = self.anchor_loss_gcl(model, anchor_adj, features, node_anchor_adj)
                semi_loss, train_accu, train_f1_macro, train_f1_micro, train_auc = self.node_anchor_loss_cls_auc(classifier, anchor_adj, features, node_anchor_adj, train_mask, labels, args.emb_fusion_ratio)
                cur_anchor_adj = compute_anchor_adj(node_anchor_adj)
                semi_loss += add_graph_degree_loss(cur_anchor_adj, args.anchor_weight)


                if args.head_tail_mi:
                    mi_loss = self.anchor_loss_gcl(model, anchor_adj, features, node_anchor_adj)
                    final_loss = semi_loss + mi_loss * args.mi_ratio
                else:
                    final_loss = semi_loss
                

                optimizer_cl.zero_grad()
                optimizer_learner.zero_grad()
                optimizer_classifer.zero_grad()

                final_loss.backward()

                optimizer_cl.step()
                optimizer_learner.step()
                optimizer_classifer.step()

                if epoch % args.eval_freq == 0:
                    classifier.eval()

                    val_loss, val_accu, val_f1_macro, val_f1_micro, val_auc = self.node_anchor_loss_cls_auc(classifier, anchor_adj, features, node_anchor_adj, val_mask, labels, args.emb_fusion_ratio)
                    test_loss, test_accu, test_f1_macro, test_f1_micro, test_auc = self.node_anchor_loss_cls_auc(classifier, anchor_adj, features, node_anchor_adj, test_mask, labels, args.emb_fusion_ratio)


                    if best_loss >= val_loss and val_accu >= best_val:
                        print('--------------- update!----------------')
                        bad_counter = 0
                        best_epoch = epoch
                        best_loss = val_loss
                        best_val = val_accu
                        best_auc = val_auc
                        best_macro_f1 = val_f1_macro
                        best_classifier = copy.deepcopy(classifier)
                        best_node_anchor_adj = node_anchor_adj
                    else:
                        bad_counter += 1

                    if bad_counter >= args.patience_cls:
                        break


                    print("Epoch {:05d} |MI Loss {:.4f} | Eval Loss {:.4f} | Eval ACC {:.4f} | Test ACC {:.4f}| Eval Macro-F1 {:.4f} | Test Macro-F1 {:.4f}| Eval AUC  {:.4f} | Test AUC {:.4f}".format(
                        epoch, mi_loss.item(), val_loss.item(), val_accu.item(), test_accu.item(), val_f1_macro.item(), test_f1_macro.item(), val_auc.item(), test_auc.item()))

            
            best_classifier.eval()
            test_loss, test_accu, test_f1_macro, test_f1_micro, test_auc = self.node_anchor_loss_cls_auc(best_classifier, anchor_adj, features, best_node_anchor_adj, test_mask, labels, args.emb_fusion_ratio)
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
                        choices=['pubmed', 'ogbn-arxiv', 'Coauthor-CS', 'Coauthor-Phy'])
    parser.add_argument('-ntrials', type=int, default=5)
    parser.add_argument('-seeds', nargs='+', type=list, default=[0,1,2,3,4])
    parser.add_argument('-sparse', type=int, default=1)
    parser.add_argument('-eval_freq', type=int, default=1)
    parser.add_argument('-epochs', type=int, default=500)
    parser.add_argument('-gpu', type=int, default=0)
    
    parser.add_argument('-preprocess', type=int, default=1)
    
    # GSL Module
    parser.add_argument('-anchor_num', type=int, default=700)
    parser.add_argument('-anchor_weight', type=float, default=0.03)
    parser.add_argument('-head_num', type=int, default=6)

    parser.add_argument('-epsilon', type=float, default=0.1)
    parser.add_argument('-emb_fusion_ratio', type=float, default=0.35)
    parser.add_argument('-anchor_adj_fusion_ratio', type=float, default=0.95)

    parser.add_argument('-stage_ks', nargs='+', type=list, default=[])
    parser.add_argument('-split_deep', type=int, default=1)
    parser.add_argument('-split_prop', type=float, default=0.7)
    
    parser.add_argument('-lr_gsl', type=float, default=0.01)
    parser.add_argument('-w_decay_gsl', type=float, default=0.0005)


    # GCL Module
    parser.add_argument('-head_tail_mi', type=int, default=1)
    parser.add_argument('-mi_ratio', type=float, default=0.1)

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
    parser.add_argument('-bn_cls', type=int, default=0)
    parser.add_argument('-lr_cls', type=float, default=0.01)
    parser.add_argument('-w_decay_cls', type=float, default=0.0005)
    parser.add_argument('-hidden_dim_cls', type=int, default=64)
    parser.add_argument('-dropout_cls', type=float, default=0.5)
    parser.add_argument('-dropedge_cls', type=float, default=0.0)
    parser.add_argument('-nlayers_cls', type=int, default=2)
    parser.add_argument('-patience_cls', type=int, default=100)

    args = parser.parse_args()

    if args.split_deep>0 and args.split_prop>0:
        args.stage_ks = [args.split_prop] * args.split_deep
        
    experiment = Experiment()
    experiment.train(args)
