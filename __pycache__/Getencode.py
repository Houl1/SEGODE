import argparse
import csv
import os
import numpy
import pandas as pd
from itertools import islice
import networkx as nx
import torch

import GetGDV
import functools
print = functools.partial(print, flush=True) #刷新print函数的输出
parser = argparse.ArgumentParser('Processing data')
parser.add_argument('--nodefile', type=str)
parser.add_argument('--inputpath', type=str)  #快照文件路径
parser.add_argument('--finalpath', type=str) #输出文件路径
parser.add_argument('--outputfile', type=str) #输出文件名
parser.add_argument('--method', type=str, choices=['dataformat', 'getvetor'])
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()
if args.gpu >= 0:
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

def dataformat(nodefile,inputpath,outputpath,outputfile):
    list_edge = []
    f_node = csv.reader(open(nodefile))
    nodes=[]
    for row in f_node:
        nodes.append(row[0])
    for (root, dirs, files) in os.walk(inputpath):
        for i, file in enumerate(files):
            f_edge = csv.reader(open(os.path.join(inputpath, file)))
            for j in islice(f_edge, 1, None):
                edge=j[0].split('\t')
                list_edge_t = []
                list_edge_t.append(nodes.index(edge[0]))
                list_edge_t.append(nodes.index(edge[1]))
                list_edge_t.append(i)
                list_edge.append(list_edge_t)
    with open(os.path.join(outputpath, outputfile),'w',newline='') as outfile:
        writer = csv.writer(outfile)
        for row in list_edge:
            writer.writerow(row)


def getvetor(filepath):
    graph_list=[]
    graph_file = csv.reader(open(os.path.join(filepath, "edges.csv")))
    head = next(graph_file)
    for edge in graph_file:
        if edge[2] != '0':
            break
        tupl = (int(edge[0]),int(edge[1]))
        graph_list.append(tupl)
    G = nx.Graph()
    G.add_nodes_from([x for x in range(int(head[0]))])
    G.add_edges_from(graph_list)

    pr_dict = nx.pagerank(G)
    pr_list = []
    for i in range(G.number_of_nodes()):
        pr_list.append([pr_dict[i]])
    pr = torch.FloatTensor(pr_list)
    print(pr.shape)

    A =  torch.FloatTensor(nx.adjacency_matrix(G).todense())
    print(A.shape)

    GDV = torch.FloatTensor(GetGDV.getresult(G, filepath))
    print((GDV.shape))
    encode = torch.cat([A,pr,GDV],dim=1)
    print(encode.shape)



if __name__ == '__main__':
    if args.method==dataformat:
        dataformat(args.nodefile, args.inputpath, args.finalpath, args.outputfile).to(device)
    elif args.method==getvetor:
        getvetor(args.finalpath).to(device)

#  --nodefile enron/nodes_set/nodes.csv --inputpath enron/1.format --finalpath enron/final --outputfile edges.csv --method dataformat --gpu 1
# --finalpath uci/final --method getvetor --gpu 1