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
import scipy.sparse as sp
from torch.utils.data import DataLoader

from utils.preprocess import load_graphs, get_context_pairs, get_evaluation_data
from utils.minibatch import DynamicsDataset
from utils.utilities import *
from eval.link_prediction import evaluate_classifier
from models.model import DynamicsModel, GeneDynamics, HeatDiffusion, MutualDynamics
import torchdiffeq as ode

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import random
import os
import wandb
wandb.init(project="myproject", entity="houjinlin" )
torch.autograd.set_detect_anomaly(True)


# python train.py --time_steps 6 --dataset uci --gpu 1 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, nargs='?', default=100,
                        help="total time steps used for train, eval and test")
    # Experimental settings.
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, nargs='?', default=2000,
                        help='# epochs')
    parser.add_argument('--test_freq', type=int, default=20)
    parser.add_argument('--batch_size', type=int, nargs='?', default=1024,
                        help='Batch size (# nodes)')
    parser.add_argument("--early_stop", type=int, default=30,
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
    parser.add_argument('--n', type=int, default=400, help='Number of nodes')
    parser.add_argument('--T', type=float, default=5., help='Terminal Time')
    parser.add_argument('--layout', type=str, choices=['community', 'degree'], default='community')
    parser.add_argument('--network', type=str,
                    choices=['grid', 'random', 'power_law', 'small_world', 'community'], default='grid')
    parser.add_argument('--physical_equation', type=str, default='gene',choices=['gene','heat','mutualistic'])

    # Architecture params
    parser.add_argument('--encoding_layer_config', type=str, nargs='?', default='32',
                        help='Encoder layer config. ')
    parser.add_argument('--adjoint', default='True')
    parser.add_argument('--graphloss_rate', default=1.0)
    parser.add_argument('--tasktype', type=str, default="siglestep",choices=['siglestep','multisteps','data_scarce'])
    parser.add_argument('--scare_snapshot', type=str, default='')

    args = parser.parse_args()
    wandb.config.update(args)
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

    
    # Build network # A: Adjacency matrix, L: Laplacian Matrix,  OM: Base Operator
    n = args.n  # e.g nodes number 400
    N = int(np.ceil(np.sqrt(n)))  # grid-layout pixels :20
    
    if args.network == 'grid':
        print("Choose graph: " + args.network)
        A = grid_8_neighbor_graph(N)
        G = nx.from_numpy_array(A.numpy())
    elif args.network == 'random':
        print("Choose graph: " + args.network)
        G = nx.erdos_renyi_graph(n, 0.1, seed=args.seed)
        G = networkx_reorder_nodes(G, args.layout)
        A = torch.FloatTensor(nx.to_numpy_array(G))
    elif args.network == 'power_law':
        print("Choose graph: " + args.network)
        G = nx.barabasi_albert_graph(n, 5, seed=args.seed)
        G = networkx_reorder_nodes(G,  args.layout)
        A = torch.FloatTensor(nx.to_numpy_array(G))
    elif args.network == 'small_world':
        print("Choose graph: " + args.network)
        G = nx.newman_watts_strogatz_graph(400, 5, 0.5, seed=args.seed)
        G = networkx_reorder_nodes(G, args.layout)
        A = torch.FloatTensor(nx.to_numpy_array(G))
    elif args.network == 'community':
        print("Choose graph: " + args.network)
        n1 = int(n/3)
        n2 = int(n/3)
        n3 = int(n/4)
        n4 = n - n1 - n2 -n3
        G = nx.random_partition_graph([n1, n2, n3, n4], .25, .01, seed=args.seed)
        G = networkx_reorder_nodes(G, args.layout)
        A = torch.FloatTensor(nx.to_numpy_array(G))    
    
    D = torch.diag(A.sum(1))
    L = (D - A)
    
    t = torch.linspace(0., args.T, args.time_steps)  # args.time_tick) # 100 vector
    # train_deli = 80
    id_train = list(range(int(args.time_steps * 0.8))) # first 80 % for train
    id_test = list(range(int(args.time_steps * 0.8), args.time_steps)) # last 20 % for test (extrapolation)
    t_train = t[id_train]
    t_test = t[id_test]
    
    # Initial Value
    x0 = torch.zeros(N, N) 
    x0[int(0.05*N):int(0.25*N), int(0.05*N):int(0.25*N)] = 25  # x0[1:5, 1:5] = 25  for N = 20 or n= 400 case
    x0[int(0.45*N):int(0.75*N), int(0.45*N):int(0.75*N)] = 20  # x0[9:15, 9:15] = 20 for N = 20 or n= 400 case
    x0[int(0.05*N):int(0.25*N), int(0.35*N):int(0.65*N)] = 17  # x0[1:5, 7:13] = 17 for N = 20 or n= 400 case
    x0 = x0.view(-1, 1).float() 
    energy = x0.sum()
    
    with torch.no_grad():
        if args.physical_equation == 'gene':
            solution_numerical = ode.odeint(GeneDynamics(A, 1), x0, t, method='dopri5')  
        elif args.physical_equation == 'heat':
            solution_numerical = ode.odeint(HeatDiffusion(L, 1), x0, t, method='dopri5')
        elif args.physical_equation == 'mutualistic':
            solution_numerical = ode.odeint(MutualDynamics(A), x0, t, method='dopri5')
        # print(solution_numerical.shape)

    true_y = solution_numerical.squeeze().t().to(device)  # 100 * 1 * 400  --squeeze--> 100 * 400 -t-> 400 * 100
    true_y0 = x0.to(device)  # 400 * 1
    true_y_train = true_y[:, id_train].to(device)  # 400*80  for train
    true_y_test = true_y[:, id_test].to(device)  # 400*20  for extrapolation prediction

    
    gdv = []
    read = open("data/{}/{}/{}".format(args.network, "/gdv", "gdv" + str(args.seed)))
    for x in read:
        tupl = x.split(" ")
        tupl = [int(t) for t in tupl]
        gdv.append(tupl)
    gdv = sp.csr_matrix(gdv, dtype=np.float32)
    
    # build dataloader and model
    dataset = DynamicsDataset(args, G, A, gdv, x0)
    dataloader = DataLoader(dataset,
                            batch_size=args.n,
                            shuffle=True,
                            num_workers=2,
                            collate_fn=DynamicsDataset.collate_fn)
    model = DynamicsModel(args, num_feat=x0.shape[1], num_gdv=gdv.shape[1], time_length=args.time_steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,eps=1e-5)
    # in training
    best_epoch_val = 0
    patient = 0
    best_epoch_loss = float("inf")
    best_epoch_relative_loss = float("inf")
    criterion = F.l1_loss  # F.mse_loss(pred_y, true_y)
    for epoch in range(args.epochs+1):
        model.train()
        for idx, feed_dict in enumerate(dataloader):
            feed_dict = to_device2(feed_dict, device)
            opt.zero_grad()
            pred_y = model(feed_dict["graph"],t_train)
            # print(pred_y.shape)
            pred_y = pred_y.squeeze().t()
            loss_train = criterion(pred_y, true_y_train)
            relative_loss_train = criterion(pred_y, true_y_train) / true_y_train.mean()
            # print(loss_train)
            loss_train.backward()
            opt.step()
        
            if epoch % args.test_freq == 0:
                with torch.no_grad():
                    pred_y = model(feed_dict["graph"], t).squeeze().t()  # odeint(model, true_y0, t)
                    loss = criterion(pred_y[:, id_test], true_y_test)
                    relative_loss = criterion(pred_y[:, id_test], true_y_test) / true_y_test.mean()

                    if loss<best_epoch_loss:
                        best_epoch_loss = loss
                    if relative_loss<best_epoch_relative_loss:
                        best_epoch_relative_loss = relative_loss
                    
                    print('Epoch {:04d}| Train Loss {:.6f}({:.6f} Relative) '
                            '| Test Loss {:.6f}({:.6f} Relative) '
                            .format(epoch, loss_train.item(), relative_loss_train.item(),
                                    loss.item(), relative_loss.item()))
                    wandb.log({"Epoch": epoch,"Train Loss":loss_train.item()," Train Relative":relative_loss_train.item(),
                               "Test Loss":loss.item(),"Test Relative":relative_loss.item()})
    with torch.no_grad():
        pred_y = model(feed_dict["graph"], t).squeeze().t()  # odeint(model, true_y0, t)
        loss = criterion(pred_y[:, id_test], true_y_test)
        relative_loss = criterion(pred_y[:, id_test], true_y_test) / true_y_test.mean()

        if loss<best_epoch_loss:
            best_epoch_loss = loss
        if relative_loss<best_epoch_relative_loss:
            best_epoch_relative_loss = relative_loss    
            
    print("Last Loss = {:.6f} Last Relative = {:.6f} ".format(best_epoch_loss,best_epoch_relative_loss))
    wandb.log({"Last Loss":best_epoch_loss,"Last Relative":best_epoch_relative_loss})










