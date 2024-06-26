
import numpy as np
import copy
import networkx as nx
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from utils.random_walk import Graph_RandomWalk
import scipy.sparse as sp
from networkx.algorithms import community
import torch


"""Random walk-based pair generation."""

def run_random_walks_n2v(graph, adj, num_walks, walk_len):
    """ In: Graph and list of nodes
        Out: (target, context) pairs from random walk sampling using 
        the sampling strategy of node2vec (deepwalk)"""
    nx_G = nx.Graph()
    for e in graph.edges():
        nx_G.add_edge(e[0], e[1])
    for edge in graph.edges():
        nx_G[edge[0]][edge[1]]['weight'] = adj[edge[0], edge[1]]

    G = Graph_RandomWalk(nx_G, False, 1.0, 1.0)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_len)
    WINDOW_SIZE = 10
    pairs = defaultdict(list)
    pairs_cnt = 0
    for walk in walks:
        for word_index, word in enumerate(walk):
            for nb_word in walk[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(walk)) + 1]:
                if nb_word != word:
                    pairs[word].append(nb_word)
                    pairs_cnt += 1
    print("# nodes with random walk samples: {}".format(len(pairs)))
    print("# sampled pairs: {}".format(pairs_cnt))
    return pairs

def fixed_unigram_candidate_sampler(true_clasees, 
                                    num_true, 
                                    num_sampled, 
                                    unique,  
                                    distortion, 
                                    unigrams):
    # TODO: implementate distortion to unigrams
    assert true_clasees.shape[1] == num_true
    samples = []
    for i in range(true_clasees.shape[0]):
        dist = copy.deepcopy(unigrams)
        candidate = list(range(len(dist)))
        taboo = true_clasees[i].cpu().tolist()
        for tabo in sorted(taboo, reverse=True):
            candidate.remove(tabo)
            dist.pop(tabo)
        sample = np.random.choice(candidate, size=num_sampled, replace=unique, p=dist/np.sum(dist))
        samples.append(sample)
    return samples

def to_device(batch, device):
    feed_dict = copy.deepcopy(batch)
    node_1, node_2, node_2_negative, graphs = feed_dict.values()
    # to device
    feed_dict["node_1"] = [x.to(device) for x in node_1]
    feed_dict["node_2"] = [x.to(device) for x in node_2]
    feed_dict["node_2_neg"] = [x.to(device) for x in node_2_negative]
    # feed_dict["adjs"] = [a.to(device) for a in adjs]
    # feed_dict["gdvs"] = [g.to(device) for g in gdvs]
    # feed_dict["prs"] = [p.to(device) for p in prs]
    feed_dict["graphs"] = [g.to(device) for g in graphs]

    return feed_dict

def to_device2(batch, device):
    feed_dict = copy.deepcopy(batch)
    graph = feed_dict["graph"]
    # to device
    feed_dict["graph"] = graph.to(device)

    return feed_dict

def get_pos_emb(pos, hid_dim):
    '''
    Funciton to get positional embedding
    :param pos: position index
    :param hid_dim: dimensionality of positional embedding
    :return: positional embedding
    '''
    pos_emb = np.zeros((1, hid_dim))
    for i in range(hid_dim):
        if i%2==0:
            pos_emb[0, i] = np.sin(pos/(10000**(i/hid_dim)))
        else:
            pos_emb[0, i] = np.cos(pos/(10000**((i-1)/hid_dim)))
    return pos_emb


def generate_node_mapping(G, type=None):
    """
    :param G:
    :param type:
    :return:
    """
    if type == 'degree':
        s = sorted(G.degree, key=lambda x: x[1], reverse=True)
        new_map = {s[i][0]: i for i in range(len(s))}
    elif type == 'community':
        cs = list(community.greedy_modularity_communities(G))
        l = []
        for c in cs:
            l += list(c)
        new_map = {l[i]:i for i in range(len(l))}
    else:
        new_map = None

    return new_map


def networkx_reorder_nodes(G, type=None):
    """
    :param G:  networkX only adjacency matrix without attrs
    :param nodes_map:  nodes mapping dictionary
    :return:
    """
    nodes_map = generate_node_mapping(G, type)
    if nodes_map is None:
        return G
    C = nx.to_scipy_sparse_matrix(G, format='coo')
    new_row = np.array([nodes_map[x] for x in C.row], dtype=np.int32)
    new_col = np.array([nodes_map[x] for x in C.col], dtype=np.int32)
    new_C = sp.coo_matrix((C.data, (new_row, new_col)), shape=C.shape)
    new_G = nx.from_scipy_sparse_matrix(new_C)
    return new_G

def grid_8_neighbor_graph(N):
    """
    Build discrete grid graph, each node has 8 neighbors
    :param n:  sqrt of the number of nodes
    :return:  A, the adjacency matrix
    """
    N = int(N)
    n = int(N ** 2)
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    A = torch.zeros(n, n)
    for x in range(N):
        for y in range(N):
            index = x * N + y
            for i in range(len(dx)):
                newx = x + dx[i]
                newy = y + dy[i]
                if N > newx >= 0 and N > newy >= 0:
                    index2 = newx * N + newy
                    A[index, index2] = 1
    return A.float()
