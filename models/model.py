# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2022/09/27
@Author  :   JinLin Hou
@Contact :   houjinlin@tju.edu.cn
'''
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss
import scipy.sparse as sp
import numpy as np
import torch_geometric as tg

from models.layers import EncodingLayer, ODEBlock, DecodingLayer,GODEEncodingLayer,DynamicsODEBlock,DynamicsDecodingLayer
from utils.utilities import fixed_unigram_candidate_sampler


from utils import *


class Mymodel(nn.Module):
    def __init__(self, args, num_feat, num_gdv, time_length):
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(Mymodel, self).__init__()
        self.args = args
        if args.tasktype == "multisteps":
            self.num_time_steps = time_length-5
        else:
            self.num_time_steps = time_length-1
        self.num_feat = num_feat
        self.num_gdv = num_gdv
        self.num_pr = 1

        self.encoding_layer_config = int(args.encoding_layer_config)
        self.time_steps = [t for t in range(time_length)]
        self.encode_drop = args.encode_drop
        self.ode_drop = args.ode_drop
        self.rtol = args.rtol
        self.atol = args.atol
        self.method = args.method
        self.adjoint = args.adjoint

        self.encoding, self.ode, self.decoding = self.build_model()
        self.init_params()


    def forward(self, graphs):
        # encoding  forward
        encoding_out = []
        for t in range(self.num_time_steps):
            encoding_out.append(self.encoding(graphs[t]))
        encoding_out = torch.stack([e for e in encoding_out] ).to(encoding_out[0].device)    

        # ODE forward
        ode_out = self.ode(encoding_out)

        #decoding forward
        decoding_out_adj = []
        for ode in ode_out:
            decode_adj = self.decoding(ode)
            decoding_out_adj.append(decode_adj)
        decoding_out_adj = torch.stack([a for a in decoding_out_adj])
        return decoding_out_adj


    def build_model(self):
        # 1:Encoding Layers
        encoding_layer = EncodingLayer(input_dim_feat=self.num_feat,
                                        input_dim_gdv=self.num_gdv,
                                        input_dim_pr=self.num_pr,
                                        output_dim=self.encoding_layer_config,
                                        drop=self.encode_drop)
        # 2:ODE Layers
        ode_layer = ODEBlock(self.args,
                            encoding_size=self.encoding_layer_config,
                            time_steps=self.time_steps,
                            dropout=self.ode_drop,
                            rtol=self.rtol,
                            atol=self.atol,
                            method=self.method,
                            adjoint =self.adjoint)

        #: Decoding Layers
        decoding_layer = DecodingLayer(input_dim = self.encoding_layer_config)
        
        return encoding_layer, ode_layer, decoding_layer



    def init_params(self):
        self.encoding.reset_parameters()
        self.decoding.reset_parameters()
        
    
    def get_loss(self, feed_dict):
        node_1, node_2, node_2_negative, graphs = feed_dict.values()
        # run gnn

        final_emb = self.forward(graphs)
        self.adj_loss1 = 0.0
        self.graph_loss4 = 0.0


        for t in range(self.num_time_steps):
            
            emb_t = final_emb[t]
            source_node_emb = emb_t[node_1[t]]
            tart_node_pos_emb = emb_t[node_2[t]]
            tart_node_neg_emb = emb_t[node_2_negative[t]]

            pos_bceloss = nn.BCELoss()
            neg_bceloss = nn.BCELoss()
                                  
            pos_score = torch.sum(source_node_emb*tart_node_pos_emb, dim=1)
            neg_score = -torch.sum(source_node_emb[ :,None , :]*tart_node_neg_emb, dim=2).flatten()
                        
            pos_score = torch.sigmoid(pos_score)
            neg_score = torch.sigmoid(neg_score)
            
            pos_loss = pos_bceloss(pos_score, torch.ones_like(pos_score))
            neg_loss = neg_bceloss(neg_score, torch.ones_like(neg_score))

            adj_loss1 = pos_loss + self.args.neg_weight * neg_loss
            self.adj_loss1 += adj_loss1
            
            # MSELoss(t-1≈t)
            L1Loss = nn.L1Loss()
            if t < self.num_time_steps-1:
                graph_loss4 = L1Loss(final_emb[2][t], final_emb[2][t+1])
                self.graph_loss4 += graph_loss4
            
        self.adj_loss1 = self.adj_loss1    
        self.graph_loss4 = self.graph_loss4 * float(self.args.graphloss_rate)                   

        return self.adj_loss1 + self.graph_loss4


class DynamicsModel(nn.Module):
    def __init__(self, args, num_feat, num_gdv, time_length):
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(DynamicsModel, self).__init__()
        self.args = args
        self.num_feat = num_feat
        self.num_gdv = num_gdv
        self.num_pr = 1

        self.encoding_layer_config = int(args.encoding_layer_config)
        self.time_steps = [t for t in range(time_length)]
        self.encode_drop = args.encode_drop
        self.ode_drop = args.ode_drop
        self.rtol = args.rtol
        self.atol = args.atol
        self.method = args.method
        self.adjoint = args.adjoint

        self.encoding, self.ode, self.decoding = self.build_model()
        self.init_params()


    def forward(self, graph, t):
        # encoding  forward
        encoding_out = []
        for i in range(len(t)):
            encoding_out.append(self.encoding(graph))
        encoding_out = torch.stack([e for e in encoding_out] ).to(encoding_out[0].device)

    
        # ODE forward
        ode_out = self.ode(encoding_out,t)

        #decoding forward
        decoding_out = []
        for ode in ode_out:
            decode = self.decoding(ode)
            decoding_out.append(decode)
        decoding_out = torch.stack([a for a in decoding_out])
        return decoding_out


    def build_model(self):
        # 1:Encoding Layers
        encoding_layer = EncodingLayer(input_dim_feat=self.num_feat,
                                        input_dim_gdv=self.num_gdv,
                                        input_dim_pr=self.num_pr,
                                        output_dim=self.encoding_layer_config,
                                        drop=self.encode_drop)
        # 2:ODE Layers
        ode_layer = DynamicsODEBlock(self.args,
                            encoding_size=self.encoding_layer_config,
                            time_steps=self.time_steps,
                            dropout=self.ode_drop,
                            rtol=self.rtol,
                            atol=self.atol,
                            method=self.method,
                            adjoint =self.adjoint)

        #: Decoding Layers
        decoding_layer = DynamicsDecodingLayer(input_dim = self.encoding_layer_config, output_dim = 1)
        
        return encoding_layer, ode_layer, decoding_layer



    def init_params(self):
        self.encoding.reset_parameters()
        self.decoding.reset_parameters()
        

        

        
class GODE(nn.Module):
    def __init__(self, args, num_feat, num_gdv, time_length):
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(GODE, self).__init__()
        self.args = args
        if args.tasktype == "multisteps":
            self.num_time_steps = time_length-5
        else:
            self.num_time_steps = time_length-1

        self.num_feat = num_feat
        self.num_gdv = num_gdv
        self.num_pr = 1

        self.encoding_layer_config = int(args.encoding_layer_config)
        self.time_steps = [t for t in range(time_length)]
        self.encode_drop = args.encode_drop
        self.ode_drop = args.ode_drop
        self.rtol = args.rtol
        self.atol = args.atol
        self.method = args.method
        self.adjoint = args.adjoint

        self.encoding, self.ode, self.decoding = self.build_model()
        self.init_params()

        

    # def forward(self, adjs, gdvs, prs):
    def forward(self, graphs):
        # encoding  forward
        encoding_out = []
        for t in range(self.num_time_steps):
            encoding_out.append(self.encoding(graphs[t]))
        encoding_out = torch.stack([e for e in encoding_out] ).to(encoding_out[0].device)    

        # ODE forward
        ode_out = self.ode(encoding_out)

        # decoding forward
        decoding_out_adj = []
        for ode in ode_out:
            decode_adj = self.decoding(ode)
            decoding_out_adj.append(decode_adj)
        decoding_out_adj = torch.stack([a for a in decoding_out_adj])
        return decoding_out_adj

    def build_model(self):
        # 1:Encoding Layers
        encoding_layer = GODEEncodingLayer(input_dim_feat=self.num_feat,
                                        input_dim_gdv=self.num_gdv,
                                        input_dim_pr=self.num_pr,
                                        output_dim=self.encoding_layer_config,
                                        drop=self.encode_drop)
        # 2:ODE Layers
        ode_layer = ODEBlock(self.args,
                            encoding_size=self.encoding_layer_config,
                            time_steps=self.time_steps,
                            dropout=self.ode_drop,
                            rtol=self.rtol,
                            atol=self.atol,
                            method=self.method,
                            adjoint =self.adjoint)

        #: Decoding Layers
        decoding_layer = DecodingLayer(input_dim=self.encoding_layer_config)

        return encoding_layer, ode_layer, decoding_layer

    def init_params(self):
        self.encoding.reset_parameters()
        self.decoding.reset_parameters()

    def get_loss(self, feed_dict):
        node_1, node_2, node_2_negative, graphs = feed_dict.values()
        # run gnn

        final_emb = self.forward(graphs)
        self.adj_loss1 = 0.0
        self.graph_loss4 = 0.0

        for t in range(self.num_time_steps):

            emb_t = final_emb[t]
            source_node_emb = emb_t[node_1[t]]
            tart_node_pos_emb = emb_t[node_2[t]]
            tart_node_neg_emb = emb_t[node_2_negative[t]]

            pos_bceloss = nn.BCELoss()
            neg_bceloss = nn.BCELoss()

            pos_score = torch.sum(source_node_emb * tart_node_pos_emb, dim=1)
            neg_score = -torch.sum(source_node_emb[:, None, :] * tart_node_neg_emb, dim=2).flatten()

            pos_score = torch.sigmoid(pos_score)
            neg_score = torch.sigmoid(neg_score)

            pos_loss = pos_bceloss(pos_score, torch.ones_like(pos_score))
            neg_loss = neg_bceloss(neg_score, torch.ones_like(neg_score))

            adj_loss1 = pos_loss + self.args.neg_weight * neg_loss
            self.adj_loss1 += adj_loss1

            # MSELoss(t-1≈t)
            L1Loss = nn.L1Loss()
            if t < self.num_time_steps - 1:
                graph_loss4 = L1Loss(final_emb[2][t], final_emb[2][t + 1])
                self.graph_loss4 += graph_loss4

        self.adj_loss1 = self.adj_loss1
        self.graph_loss4 = self.graph_loss4 * float(self.args.graphloss_rate)

        return self.adj_loss1 + self.graph_loss4


class GeneDynamics(nn.Module):
    def __init__(self,  A,  b, f=1, h=2):
        super(GeneDynamics, self).__init__()
        self.A = A   # Adjacency matrix
        self.b = b
        self.f = f
        self.h = h

    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dxi(t)/dt = -b*xi^f + \sum_{j=1}^{N}Aij xj^h / (1 + xj^h)
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
            f = -self.b * (x ** self.f) + torch.sparse.mm(self.A, x**self.h / (x**self.h + 1))
        else:
            f = -self.b * (x ** self.f) + torch.mm(self.A, x ** self.h / (x ** self.h + 1))
        return f

class HeatDiffusion(nn.Module):
    # In this code, row vector: y'^T = y^T A^T      textbook format: column vector y' = A y
    def __init__(self,  L,  k=1):
        super(HeatDiffusion, self).__init__()
        self.L = -L  # Diffusion operator
        self.k = k   # heat capacity

    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dX(t)/dt = -k * L *X
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        if hasattr(self.L, 'is_sparse') and self.L.is_sparse:
            f = torch.sparse.mm(self.L, x)
        else:
            f = torch.mm(self.L, x)
        return self.k * f


class MutualDynamics(nn.Module):
    #  dx/dt = b +
    def __init__(self, A, b=0.1, k=5., c=1., d=5., e=0.9, h=0.1):
        super(MutualDynamics, self).__init__()
        self.A = A   # Adjacency matrix, symmetric
        self.b = b
        self.k = k
        self.c = c
        self.d = d
        self.e = e
        self.h = h

    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dxi(t)/dt = bi + xi(1-xi/ki)(xi/ci-1) + \sum_{j=1}^{N}Aij *xi *xj/(di +ei*xi + hi*xj)
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        n, d = x.shape
        f = self.b + x * (1 - x/self.k) * (x/self.c - 1)
        if d == 1:
            # one 1 dim can be computed by matrix form
            if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
                outer = torch.sparse.mm(self.A,
                                        torch.mm(x, x.t()) / (self.d + (self.e * x).repeat(1, n) + (self.h * x.t()).repeat(n, 1)))
            else:
                outer = torch.mm(self.A,
                                 torch.mm(x, x.t()) / (
                                             self.d + (self.e * x).repeat(1, n) + (self.h * x.t()).repeat(n, 1)))
            f += torch.diag(outer).view(-1, 1)
        else:
            # high dim feature, slow iteration
            if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
                vindex = self.A._indices().t()
                for k in range(self.A._values().__len__()):
                    i = vindex[k, 0]
                    j = vindex[k, 1]
                    aij = self.A._values()[k]
                    f[i] += aij * (x[i] * x[j]) / (self.d + self.e * x[i] + self.h * x[j])
            else:
                vindex = self.A.nonzero()
                for index in vindex:
                    i = index[0]
                    j = index[1]
                    f[i] += self.A[i, j]*(x[i] * x[j]) / (self.d + self.e * x[i] + self.h * x[j])
        return f