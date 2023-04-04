# -*- encoding: utf-8 -*-
'''
@File    :   layers.py
@Time    :   2022/09/27
@Author  :   JinLin Hou
@Contact :   houjinlin@tju.edu.cn
'''
from utils.utilities import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
import torchdiffeq as ode
import torch_geometric as tg
import numpy as np
import copy
import math

class EncodingLayer(nn.Module):
    def __init__(self,
                 input_dim_feat,
                 output_dim,
                 input_dim_gdv=73,
                 input_dim_pr=1,
                 drop=0.0,
                 ):
        super(EncodingLayer, self).__init__()
        self.output_dim = output_dim
        self.drop = drop

        self.input_dim_feat = input_dim_feat
        self.input_dim_gdv = input_dim_gdv
        self.input_dim_pr = input_dim_pr


        self.gdv_layer = nn.Sequential(nn.Linear(input_dim_gdv, output_dim//4, bias=True),
                                      nn.Sigmoid(),nn.Dropout(drop))
        self.pr_layer = nn.Sequential(nn.Linear(input_dim_pr, output_dim//4, bias=True),
                                      nn.Sigmoid(),nn.Dropout(drop))
        self.gcn_layer1 = GCNConv(input_dim_feat, output_dim, dropout=drop)
        self.gcn_layer2 = GCNConv(output_dim, output_dim//2, dropout=drop)
        
        self.encode_layer = nn.Sequential(nn.Linear(output_dim, output_dim, bias=True), nn.Tanh(),
                               nn.Linear(output_dim, output_dim, bias=True))

    def forward(self, graph):
        graph = copy.deepcopy(graph)
        edge_index = graph.edge_index
        feat = graph.x
        gdv = graph.gdv
        pr = graph.pr
        edge_weight = graph.edge_weight

        adj_encode = self.gcn_layer1(feat, edge_index, edge_weight)
        adj_encode = torch.tanh(adj_encode)
        adj_encode = self.gcn_layer2(adj_encode, edge_index, edge_weight)
        adj_encode = torch.sigmoid(adj_encode)
        
        # GDV and PR
        gdv = self.gdv_layer(gdv)
        pr = self.pr_layer(pr)

        # Encode cat 
        encode = torch.cat([adj_encode, gdv, pr], dim=1)
        # encode = torch.cat([adj_encode, gdv], dim=1)
        encode = self.encode_layer(encode)
        return encode

    def reset_parameters(self):
        for l in self.gdv_layer:
            try:
                l.reset_parameters()
            except:
                continue
        for l in self.pr_layer:
            try:
                l.reset_parameters()
            except:
                continue
        self.gcn_layer1.reset_parameters()
        self.gcn_layer2.reset_parameters()
        for l in self.encode_layer:
            try:
                l.reset_parameters()
            except:
                continue


class GODEEncodingLayer(nn.Module):
    def __init__(self,
                 input_dim_feat,
                 output_dim,
                 input_dim_gdv=73,
                 input_dim_pr=1,
                 drop=0.0,
                 ):
        super(GODEEncodingLayer, self).__init__()
        self.output_dim = output_dim
        self.drop = drop

        self.input_dim_feat = input_dim_feat
        self.input_dim_gdv = input_dim_gdv
        self.input_dim_pr = input_dim_pr
        
        self.gcn_layer1 = GCNConv(input_dim_feat, output_dim, dropout=drop)
        self.gcn_layer2 = GCNConv(output_dim, output_dim, dropout=drop)

    def forward(self, graph):
        graph = copy.deepcopy(graph)
        edge_index = graph.edge_index
        feat = graph.x

        edge_weight = graph.edge_weight

        
        adj_encode = self.gcn_layer1(feat, edge_index, edge_weight)
        adj_encode = torch.tanh(adj_encode)
        adj_encode = self.gcn_layer2(adj_encode, edge_index, edge_weight)
        adj_encode = torch.sigmoid(adj_encode)

        return adj_encode

    def reset_parameters(self):
        self.gcn_layer1.reset_parameters()
        self.gcn_layer2.reset_parameters()

class ODEFunc(nn.Module):  # A kind of ODECell in the view of RNN
    def __init__(self,
                 args,
                 hidden_size,
                 encodings,
                 dropout=0.1):
        super(ODEFunc, self).__init__()
        self.args = args
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.timeparament = nn.Parameter(torch.Tensor(1,hidden_size))
        self.encode = encodings
        self.L1 = nn.Linear(hidden_size,hidden_size,bias=False)
        self.L2 = nn.Linear(hidden_size,hidden_size,bias=False)
        self.L3 = nn.Linear(hidden_size,hidden_size,bias=False)
        self.dropout_layer = nn.Dropout(dropout)
        if self.args.tasktype == "data_scarce":
            self.data = list(map(int, self.args.scare_snapshot.split(",")))
        
        
    def forward(self,t, x):
        if self.args.tasktype == "data_scarce":
            if int(t)==t and int(t)<self.encode.shape[0]-1 and int(t) not in self.data:
                control_gradient = self.encode[int(t)]   
            elif int(t)<self.encode.shape[0]-1:
                control_low = math.floor(t)
                control_high = math.ceil(t)
                for i in range(math.floor(t),-1,-1):
                    if i not in self.data:
                        control_low = i
                        break
                for i in range(math.ceil(t), self.encode.shape[0]):
                    if i not in self.data:
                        control_high = i
                        break
                control_gradient = (control_high-t)/(control_high-control_low)*self.encode[control_low]+(t-control_low)/(control_high-control_low)*self.encode[control_high]
            else:
                control_gradient = self.encode[-1]
        else:    
            if int(t)==t and int(t)<self.encode.shape[0]-1:    
                control_gradient = self.encode[int(t)]
            elif int(t)<self.encode.shape[0]-1:
                control_gradient = (1-(t-int(t)))*self.encode[int(t)]+(t-int(t))*self.encode[int(t)+1]
            else:   
                control_gradient = self.encode[-1]

        control_gradient.to(self.encode.device)
        out1 = self.L1(control_gradient)
        out2 = self.L2(control_gradient)
        out = nn.Softmax(dim=-1)(torch.matmul(out1,out2.t())/math.sqrt(self.hidden_size))
        out = torch.matmul(out,x)
        return out
    
    def reset_parameters(self):
        self.L1.reset_parameters()
        self.L2.reset_parameters()

class ODEBlock(nn.Module):
    def __init__(self, args,encoding_size, time_steps, dropout=.0, rtol=.01, atol=.001, method='dopri5', adjoint=True, terminal=False):
        """
        :param odefunc: X' = f(X, t, G, W)
        :param rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        :param atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        :param method:
            'explicit_adams': AdamsBashforth,
            'fixed_adams': AdamsBashforthMoulton,
            'adams': VariableCoefficientAdamsBashforth,
            'tsit5': Tsit5Solver,
            'dopri5': Dopri5Solver,
            'euler': Euler,
            'midpoint': Midpoint,
            'rk4': RK4,
        """

        super(ODEBlock, self).__init__()
        self.args = args
        self.encoding_size = encoding_size
        self.integration_time_vector = time_steps  # time vector
        self.drop = dropout
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint
        self.terminal = terminal
        self.L = nn.Linear(encoding_size,encoding_size)

    def forward(self, encodings):
        integration_time_vector = torch.FloatTensor(self.integration_time_vector).to(encodings.device)
        odefunc = self.bulid_odefunc(encodings)
        odefirst = self.L(encodings[0])
        if self.adjoint:
            out = ode.odeint_adjoint(odefunc, odefirst, integration_time_vector,
                                     rtol=self.rtol, atol=self.atol, method=self.method).to(encodings.device)
        else:
            out = ode.odeint(odefunc, odefirst, integration_time_vector,
                             rtol=self.rtol, atol=self.atol, method=self.method).to(encodings.device)
        return out    

    def bulid_odefunc(self, encodings):
        self.func = ODEFunc(self.args,self.encoding_size, encodings=encodings,dropout=self.drop).to(encodings.device)
        return self.func
    def reset_parameters(self):
        self.L.reset_parameters()
    
class DecodingLayer(nn.Module):
    def __init__(self,
                 input_dim):
        super(DecodingLayer, self).__init__()
        self.input_dim = input_dim
        self.out_adj_layer = nn.Sequential(nn.Linear(input_dim, input_dim, bias=True), nn.Linear(input_dim, input_dim, bias=True))
        
    def forward(self, ode_out):
        adj = self.out_adj_layer(ode_out)
        return adj

    def reset_parameters(self):
        for l in self.out_adj_layer:
            try:
                l.reset_parameters()
            except:
                continue