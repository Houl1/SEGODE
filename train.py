# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2022/09/29
@Author  :   Hou Jinlin
@Contact :   1252405352@qq.com
'''
import argparse
import networkx as nx
import numpy as np
import dill
import pickle as pkl
import scipy
from torch.utils.data import DataLoader

from utils.preprocess import load_graphs, get_context_pairs, get_evaluation_data
from utils.minibatch import MyDataset
from utils.utilities import to_device
from eval.link_prediction import evaluate_classifier
from models.model import Mymodel

import torch
import torch.backends.cudnn as cudnn
import random
import os
torch.autograd.set_detect_anomaly(True)


def inductive_graph(graph_former, graph_later):
    """Create the adj_train so that it includes nodes from (t+1)
       but only edges from t: this is for the purpose of inductive testing.

    Args:
        graph_former ([type]): [description]
        graph_later ([type]): [description]
    """
    newG = nx.MultiGraph()
    newG.add_nodes_from(graph_later.nodes(data=True))
    newG.add_edges_from(graph_former.edges(data=False))
    return newG

# python train.py --time_steps 6 --dataset uci --gpu 1 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, nargs='?', default=7,
                        help="total time steps used for train, eval and test")
    # Experimental settings.
    parser.add_argument('--dataset', type=str, nargs='?', default='uci',
                        help='dataset name')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, nargs='?', default=200,
                        help='# epochs')
    parser.add_argument('--batch_size', type=int, nargs='?', default=1024,
                        help='Batch size (# nodes)')
    parser.add_argument("--early_stop", type=int, default=10,
                        help="patient")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed")

    # Tunable hyper-params
    # TODO: Implementation has not been verified, performance may not be good.
    # parser.add_argument('--residual', type=bool, nargs='?', default=True,
    #                     help='Use residual')
    # Number of negative samples per positive pair.
    parser.add_argument('--neg_sample_size', type=int, nargs='?', default=10,
                        help='# negative samples per positive')
    # Walk length for random walk sampling.
    parser.add_argument('--walk_len', type=int, nargs='?', default=20,
                        help='Walk length for random walk sampling')
    # Weight for negative samples in the binary cross-entropy loss function.
    parser.add_argument('--neg_weight', type=float, nargs='?', default=1.0,
                        help='Weightage for negative samples')
    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.01,
                        help='Initial learning rate for self-attention model.')
    parser.add_argument('--encode_drop', type=float, nargs='?', default=0.05,
                        help='Encoding Dropout (1 - keep probability).')
    parser.add_argument('--ode_drop', type=float, nargs='?', default=0.05,
                        help='ODE Dropout (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
                        help='Initial learning rate for self-attention model.')
    parser.add_argument('--method', type=str,
                        choices=["dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun", "euler", "midpoint", "rk4", "explicit_adams", "implicit_adams", "fixed_adams", "scipy_solver"],
                        default='dopri5')
    parser.add_argument('--rtol', type=float, default=0.01,
                        help='optional float64 Tensor specifying an upper bound on relative error, per element of y')
    parser.add_argument('--atol', type=float, default=0.001,
                        help='optional float64 Tensor specifying an upper bound on absolute error, per element of y')

    # Architecture params
    parser.add_argument('--encoding_layer_config', type=str, nargs='?', default='256',
                        help='Encoder layer config. ')
    parser.add_argument('--adjoint', default='True')
    parser.add_argument('--graphloss_rate', default=1.0)
    parser.add_argument('--tasktype', type=str, default="siglestep",choices=['siglestep','multisteps','data_scarce'])
    parser.add_argument('--scare_snapshot', type=str, default='')

    args = parser.parse_args()
    print(args)

    if args.gpu >= 0:
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        cudnn.deterministic = True
    setup_seed(args.seed)

    graphs, adjs, gdvs = load_graphs(args.dataset, args.time_steps)

    assert args.time_steps <= len(adjs), "Time steps is illegal"

    context_pairs_train = get_context_pairs(graphs, adjs)

    # Load evaluation data for link prediction.
    train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, \
    test_edges_pos, test_edges_neg = get_evaluation_data(graphs)
    print("No. Train: Pos={}, Neg={} \nNo. Val: Pos={}, Neg={} \nNo. Test: Pos={}, Neg={}".format(
        len(train_edges_pos), len(train_edges_neg), len(val_edges_pos), len(val_edges_neg),
        len(test_edges_pos), len(test_edges_neg)))

    # Create the adj_train so that it includes nodes from (t+1) but only edges from t: this is for the purpose of
    # inductive testing.
    new_G = inductive_graph(graphs[args.time_steps-2], graphs[args.time_steps-1])
    graphs[args.time_steps-1] = new_G
    adjs[args.time_steps-1] = nx.adjacency_matrix(new_G)

    # build dataloader and model
    dataset = MyDataset(args, graphs, adjs, gdvs, context_pairs_train)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=2,
                            collate_fn=MyDataset.collate_fn)
    model = Mymodel(args, num_feat=1, num_gdv=gdvs[0].shape[1], time_length=args.time_steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,eps=1e-5)
    # in training
    best_epoch_val = 0
    patient = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = []
        for idx, feed_dict in enumerate(dataloader):
            feed_dict = to_device(feed_dict, device)
            opt.zero_grad()
            loss = model.get_loss(feed_dict)
            loss.backward()
            opt.step()
            epoch_loss.append(loss.item())


        model.eval()
        emb = model(feed_dict["graphs"])[-1].detach().cpu().numpy()
        val_results, test_results, _, _ = evaluate_classifier(train_edges_pos,
                                                              train_edges_neg,
                                                              val_edges_pos,
                                                              val_edges_neg,
                                                              test_edges_pos,
                                                              test_edges_neg,
                                                              emb,
                                                              emb)
        epoch_auc_val = val_results["HAD"][1]
        epoch_auc_test = test_results["HAD"][1]
        
        
        if epoch_auc_val > best_epoch_val:
            best_epoch_val = epoch_auc_val
            torch.save(model.state_dict(), "./model_checkpoints/model.pt")
            patient = 0
        else:
            patient += 1
            if patient > args.early_stop:
                break

        print("Epoch {:<3},  Loss = {:.3f}, Val AUC {:.3f} Test AUC {:.3f}".format(epoch,
                                                                                   np.mean(epoch_loss),
                                                                                   epoch_auc_val,
                                                                                   epoch_auc_test))
    # Test Best Model
    model.load_state_dict(torch.load("./model_checkpoints/model.pt"))
    model.eval()
    emb = model(feed_dict["graphs"])[-1].detach().cpu().numpy()
    val_results, test_results, _, _ = evaluate_classifier(train_edges_pos,
                                                          train_edges_neg,
                                                          val_edges_pos,
                                                          val_edges_neg,
                                                          test_edges_pos,
                                                          test_edges_neg,
                                                          emb,
                                                          emb)
    auc_val = val_results["HAD"][1]
    auc_test = test_results["HAD"][1]

    print("Best Test AUC = {:.3f}".format(auc_test))









