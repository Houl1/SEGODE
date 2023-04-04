from typing import DefaultDict
from collections import defaultdict
from torch.functional import Tensor
from torch_geometric.data import Data
from utils.utilities import *
import torch
import torch_sparse
import numpy as np
import torch_geometric as tg
import scipy.sparse as sp
import networkx as nx
import scipy


import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, args, graphs, adjs, gdvs, context_pairs):
        super(MyDataset, self).__init__()
        self.args = args
        if args.tasktype == "multisteps":
            self.time_steps = args.time_steps-4
        else:
            self.time_steps = args.time_steps

        self.graphs = graphs
        self.adjs = [self._normalize_graph_gcn(a) for a in adjs]
        self.gdvs = [self._preprocess_gdvs(gdv) for gdv in gdvs]
        self.prs = self.contruct_prs()
        self.degs = self.construct_degs()
        self.feats = self.contruct_feats()
        self.context_pairs = context_pairs
        self.max_positive = args.neg_sample_size
        self.train_nodes = list(self.graphs[self.time_steps-1].nodes()) # all nodes in the graph.
        self.pyg_graphs = self._build_pyg_graphs()
        self.__createitems__()

    def _normalize_graph_gcn(self, adj):
        """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
        """
            D^(-0.5)^T * (A+I) * D^(-0.5)
        """
        adj = sp.coo_matrix(adj, dtype=np.float32)
        adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32)
        rowsum = np.array(adj_.sum(1), dtype=np.float32)
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return adj_normalized


    def _preprocess_gdvs(self, gdv):
        """Row-based normalization of GDV matrix (scipy sparse format). Output is in tuple format"""
        gdv = np.array(gdv.todense())
        return gdv

    
    def contruct_prs(self):
        """ Compute node pagerank in each graph snapshot."""
        prs = []
        for i in range(self.time_steps):
            G = self.graphs[i]
            pr_dict = nx.pagerank(G)
            pr_list = []
            for j in range(G.number_of_nodes()):
                pr_list.append([pr_dict[j]])
            pr = np.array(pr_list)
            prs.append(pr)
        return prs

    def construct_degs(self):
        """ Compute node degrees in each graph snapshot."""
        # different from the original implementation
        # degree is counted using multi graph
        degs = []
        for i in range(0, self.time_steps):
            G = self.graphs[i]
            deg = []
            for nodeid in G.nodes():
                deg.append(G.degree(nodeid))
            degs.append(deg)
        return degs
    
    def contruct_feats(self):
        feats = []
        for i in self.degs:
            x = np.array(i)
            x = x.reshape(-1,1)
            feats.append(x)
        return feats

    
    def contruct_feats_post(self):
        # Get positional embedding
        pos_emb = None
        for p in range(self.graphs[0].number_of_nodes()):
            if p==0:
                pos_emb = get_pos_emb(p, int(self.args.encoding_layer_config))
            else:
                pos_emb = np.concatenate((pos_emb, get_pos_emb(p, int(self.args.encoding_layer_config))), axis=0)
        return pos_emb

    def _build_pyg_graphs(self):
        pyg_graphs = []
        for adj, gdv, pr, feat in zip(self.adjs, self.gdvs, self.prs, self.feats):
            edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)
            feat = torch.FloatTensor(feat)
            gdv = torch.FloatTensor(gdv)
            pr = torch.FloatTensor(pr)
            data = Data(x=feat, edge_index=edge_index, edge_weight=edge_weight, gdv=gdv, pr=pr)
            pyg_graphs.append(data)
        return pyg_graphs

    def __len__(self):
        return len(self.train_nodes)

    def __getitem__(self, index):
        node = self.train_nodes[index]
        return self.data_items[node]

    def __createitems__(self):
        self.data_items = {}
        for node in list(self.graphs[self.time_steps-1].nodes()):
            feed_dict = {}
            node_1_all_time = []
            node_2_all_time = []
            for t in range(0, self.time_steps):
                node_1 = []
                node_2 = []
                if len(self.context_pairs[t][node]) > self.max_positive:
                    node_1.extend([node]* self.max_positive)
                    node_2.extend(np.random.choice(self.context_pairs[t][node], self.max_positive, replace=False))
                else:
                    node_1.extend([node]* len(self.context_pairs[t][node]))
                    node_2.extend(self.context_pairs[t][node])
                assert len(node_1) == len(node_2)
                node_1_all_time.append(node_1)
                node_2_all_time.append(node_2)

            node_1_list = [torch.LongTensor(node) for node in node_1_all_time]
            node_2_list = [torch.LongTensor(node) for node in node_2_all_time]
            node_2_negative = []
            for t in range(len(node_2_list)):
                degree = self.degs[t]
                node_positive = node_2_list[t][:, None]
                node_negative = fixed_unigram_candidate_sampler(true_clasees=node_positive,
                                                                num_true=1,
                                                                num_sampled=self.args.neg_sample_size,
                                                                unique=False,
                                                                distortion=0.75,
                                                                unigrams=degree)
                node_2_negative.append(node_negative)
            node_2_neg_list = [torch.LongTensor(np.array(node)) for node in node_2_negative]
            feed_dict['node_1']=node_1_list
            feed_dict['node_2']=node_2_list
            feed_dict['node_2_neg']=node_2_neg_list
            feed_dict["graphs"] = self.pyg_graphs

            self.data_items[node] = feed_dict

    @staticmethod
    def collate_fn(samples):
        batch_dict = {}
        for key in ["node_1", "node_2", "node_2_neg"]:
            data_list = []
            for sample in samples:
                data_list.append(sample[key])
            concate = []
            for t in range(len(data_list[0])):
                concate.append(torch.cat([data[t] for data in data_list]))
            batch_dict[key] = concate
        batch_dict["graphs"] = samples[0]["graphs"]
        return batch_dict


    
