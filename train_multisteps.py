# -*- encoding: utf-8 -*-
'''
@File    :   train_multisteps.py
@Time    :   2022/09/29
@Author  :   Jinlin Hou
@Contact :   houjinlin@tju.edu.cn
'''
import argparse
import networkx as nx
import numpy as np
import dill
import pickle as pkl
import scipy
from torch.utils.data import DataLoader

from utils.preprocess import load_graphs, get_context_pairs, get_multistep_evaluation_data
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
    parser.add_argument('--time_steps', type=int, nargs='?', default=16,
                        help="total time steps used for train, eval and test")
    # Experimental settings.
    parser.add_argument('--dataset', type=str, nargs='?', default='uci',
                        help='dataset name')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, nargs='?', default=200,
                        help='# epochs')
    parser.add_argument('--val_freq', type=int, nargs='?', default=1,
                        help='Validation frequency (in epochs)')
    parser.add_argument('--test_freq', type=int, nargs='?', default=1,
                        help='Testing frequency (in epochs)')
    parser.add_argument('--batch_size', type=int, nargs='?', default=512,
                        help='Batch size (# nodes)')
    # parser.add_argument('--featureless', type=bool, nargs='?', default=True,
    #                     help='True if one-hot encoding.')
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
                        default='dopri5')  # dopri5=explicit_adams
    #dopri8需要大内存
    parser.add_argument('--rtol', type=float, default=0.01,
                        help='optional float64 Tensor specifying an upper bound on relative error, per element of y')
    parser.add_argument('--atol', type=float, default=0.001,
                        help='optional float64 Tensor specifying an upper bound on absolute error, per element of y')

    # Architecture params
    parser.add_argument('--encoding_layer_config', type=str, nargs='?', default='256',
                        help='Encoder layer config. ')
    parser.add_argument('--adjoint', default='True')
    parser.add_argument('--graphloss_rate', default=1.0)
    parser.add_argument('--tasktype', type=str, default="siglestep", choices=['siglestep', 'multisteps', 'data_scarce'])
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

    context_pairs_train = get_context_pairs(graphs[:-4], adjs[:-4])


    # Load evaluation data for link prediction.
    train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, \
    test_edges_pos, test_edges_neg = get_multistep_evaluation_data(graphs)
    print("No. Train1: Pos={}, Neg={} | No. Val1: Pos={}, Neg={} | No. Test1: Pos={}, Neg={}".format(
        len(train_edges_pos[0]), len(train_edges_neg[0]), len(val_edges_pos[0]), len(val_edges_neg[0]),
        len(test_edges_pos[0]), len(test_edges_neg[0])))
    print("No. Train2: Pos={}, Neg={} | No. Val2: Pos={}, Neg={} | No. Test2: Pos={}, Neg={}".format(
        len(train_edges_pos[1]), len(train_edges_neg[1]), len(val_edges_pos[1]), len(val_edges_neg[1]),
        len(test_edges_pos[1]), len(test_edges_neg[1])))
    print("No. Train3: Pos={}, Neg={} | No. Val3: Pos={}, Neg={} | No. Test3: Pos={}, Neg={}".format(
        len(train_edges_pos[2]), len(train_edges_neg[2]), len(val_edges_pos[2]), len(val_edges_neg[2]),
        len(test_edges_pos[2]), len(test_edges_neg[2])))
    print("No. Train4: Pos={}, Neg={} | No. Val4: Pos={}, Neg={} | No. Test4: Pos={}, Neg={}".format(
        len(train_edges_pos[3]), len(train_edges_neg[3]), len(val_edges_pos[3]), len(val_edges_neg[3]),
        len(test_edges_pos[3]), len(test_edges_neg[3])))
    print("No. Train5: Pos={}, Neg={} | No. Val5: Pos={}, Neg={} | No. Test5: Pos={}, Neg={}".format(
        len(train_edges_pos[4]), len(train_edges_neg[4]), len(val_edges_pos[4]), len(val_edges_neg[4]),
        len(test_edges_pos[4]), len(test_edges_neg[4])))


    # build dataloader and model
    dataset = MyDataset(args, graphs[:-4], adjs[:-4], gdvs[:-4], context_pairs_train)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=2,
                            collate_fn=MyDataset.collate_fn)
    model = Mymodel(args, num_feat=1, num_gdv=gdvs[0].shape[1], time_length=args.time_steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,eps=1e-5)
    # in training
    best_epoch_val = 0
    best_epoch_test = [0,0,0,0,0]
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
        emb1 = model(feed_dict["graphs"])[-5].detach().cpu().numpy()
        val_results1, test_results1, _, _ = evaluate_classifier(train_edges_pos[0],
                                                              train_edges_neg[0],
                                                              val_edges_pos[0],
                                                              val_edges_neg[0],
                                                              test_edges_pos[0],
                                                              test_edges_neg[0],
                                                              emb1,
                                                              emb1)
        epoch_auc_val1 = val_results1["HAD"][1]
        epoch_auc_test1 = test_results1["HAD"][1]
        
        emb2 = model(feed_dict["graphs"])[-4].detach().cpu().numpy()
        val_results2, test_results2, _, _ = evaluate_classifier(train_edges_pos[1],
                                                              train_edges_neg[1],
                                                              val_edges_pos[1],
                                                              val_edges_neg[1],
                                                              test_edges_pos[1],
                                                              test_edges_neg[1],
                                                              emb2,
                                                              emb2)
        epoch_auc_val2 = val_results2["HAD"][1]
        epoch_auc_test2 = test_results2["HAD"][1]

        emb3 = model(feed_dict["graphs"])[-3].detach().cpu().numpy()
        val_results3, test_results3, _, _ = evaluate_classifier(train_edges_pos[2],
                                                              train_edges_neg[2],
                                                              val_edges_pos[2],
                                                              val_edges_neg[2],
                                                              test_edges_pos[2],
                                                              test_edges_neg[2],
                                                              emb3,
                                                              emb3)
        epoch_auc_val3 = val_results3["HAD"][1]
        epoch_auc_test3 = test_results3["HAD"][1]        

        emb4 = model(feed_dict["graphs"])[-2].detach().cpu().numpy()
        val_results4, test_results4, _, _ = evaluate_classifier(train_edges_pos[3],
                                                              train_edges_neg[3],
                                                              val_edges_pos[3],
                                                              val_edges_neg[3],
                                                              test_edges_pos[3],
                                                              test_edges_neg[3],
                                                              emb4,
                                                              emb4)
        epoch_auc_val4 = val_results4["HAD"][1]
        epoch_auc_test4 = test_results4["HAD"][1]

        emb5 = model(feed_dict["graphs"])[-4].detach().cpu().numpy()
        val_results5, test_results5, _, _ = evaluate_classifier(train_edges_pos[4],
                                                              train_edges_neg[4],
                                                              val_edges_pos[4],
                                                              val_edges_neg[4],
                                                              test_edges_pos[4],
                                                              test_edges_neg[4],
                                                              emb5,
                                                              emb5)
        epoch_auc_val5 = val_results5["HAD"][1]
        epoch_auc_test5 = test_results5["HAD"][1]

        
        if epoch_auc_val1 > best_epoch_val:
            best_epoch_val = epoch_auc_val1
            best_epoch_test = [epoch_auc_test1,epoch_auc_test2,epoch_auc_test3,epoch_auc_test4,epoch_auc_test5]
            torch.save(model.state_dict(), "./model_checkpoints/model.pt")
            patient = 0
        else:
            patient += 1
            if patient > args.early_stop:
                break

        print("Epoch {:<3},  Loss = {:.3f}, Val1 AUC {:.4f} Test1 AUC {:.4f}\n Val2 AUC {:.4f} Test2 AUC {:.4f}\n Val3 AUC {:.4f} Test3 AUC {:.4f}\n Val4 AUC {:.4f} Test4 AUC {:.4f}\n Val5 AUC {:.4f} Test5 AUC {:.4f}".format(epoch,
                                                                                   np.mean(epoch_loss),
                                                                                   epoch_auc_val1,
                                                                                   epoch_auc_test1,
                                                                                   epoch_auc_val2,
                                                                                   epoch_auc_test2,
                                                                                   epoch_auc_val3,
                                                                                   epoch_auc_test3,
                                                                                   epoch_auc_val4,
                                                                                   epoch_auc_test4,
                                                                                   epoch_auc_val5,
                                                                                   epoch_auc_test5
                                                                                   ))

    # Test Best Model
    model.load_state_dict(torch.load("./model_checkpoints/model.pt"))
    model.eval()
    emb1 = model(feed_dict["graphs"])[-5].detach().cpu().numpy()
    val_results1, test_results1, _, _ = evaluate_classifier(train_edges_pos[0],
                                                          train_edges_neg[0],
                                                          val_edges_pos[0],
                                                          val_edges_neg[0],
                                                          test_edges_pos[0],
                                                          test_edges_neg[0],
                                                          emb1,
                                                          emb1)
    auc_val1 = val_results1["HAD"][1]
    auc_test1 = test_results1["HAD"][1]

    emb2 = model(feed_dict["graphs"])[-4].detach().cpu().numpy()
    val_results2, test_results2, _, _ = evaluate_classifier(train_edges_pos[1],
                                                          train_edges_neg[1],
                                                          val_edges_pos[1],
                                                          val_edges_neg[1],
                                                          test_edges_pos[1],
                                                          test_edges_neg[1],
                                                          emb2,
                                                          emb2)
    auc_val2 = val_results2["HAD"][1]
    auc_test2 = test_results2["HAD"][1]

    emb3 = model(feed_dict["graphs"])[-3].detach().cpu().numpy()
    val_results3, test_results3, _, _ = evaluate_classifier(train_edges_pos[2],
                                                          train_edges_neg[2],
                                                          val_edges_pos[2],
                                                          val_edges_neg[2],
                                                          test_edges_pos[2],
                                                          test_edges_neg[2],
                                                          emb3,
                                                          emb3)
    auc_val3 = val_results3["HAD"][1]
    auc_test3 = test_results3["HAD"][1]

    emb4 = model(feed_dict["graphs"])[-2].detach().cpu().numpy()
    val_results4, test_results4, _, _ = evaluate_classifier(train_edges_pos[3],
                                                          train_edges_neg[3],
                                                          val_edges_pos[3],
                                                          val_edges_neg[3],
                                                          test_edges_pos[3],
                                                          test_edges_neg[3],
                                                          emb4,
                                                          emb4)
    auc_val4 = val_results4["HAD"][1]
    auc_test4 = test_results4["HAD"][1]

    emb5 = model(feed_dict["graphs"])[-4].detach().cpu().numpy()
    val_results5, test_results5, _, _ = evaluate_classifier(train_edges_pos[4],
                                                          train_edges_neg[4],
                                                          val_edges_pos[4],
                                                          val_edges_neg[4],
                                                          test_edges_pos[4],
                                                          test_edges_neg[4],
                                                          emb5,
                                                          emb5)
    auc_val5 = val_results5["HAD"][1]
    auc_test5 = test_results5["HAD"][1]

    print("Best Test1 AUC = {:.4f} | Best Test2 AUC = {:.4f} | Best Test3 AUC = {:.4f} | Best Test4 AUC = {:.4f} | Best Test5 AUC = {:.4f} |".format(auc_test1,
                                                                                                                                                     auc_test2,
                                                                                                                                                     auc_test3,
                                                                                                                                                     auc_test4,
                                                                                                                                                     auc_test5))









