import pandas as pd
import numpy as np
import networkx as nx
import cdt
import matplotlib.pyplot as plt
import argparse

import index

parser = argparse.ArgumentParser()
parser.add_argument('data', type=str)
parser.add_argument('gold', type=str)
parser.add_argument('pert', type=str)
parser.add_argument("--dual", type=int, default=0)
parser.add_argument("-p", type=int, default=0)
parser.add_argument("-v", type=int, default=0)

args = parser.parse_args()

data_file = args.gold
gold = pd.read_csv(data_file, sep="\t", header=None)


#data_file = "knockout.tsv"
data_file = args.data
data = pd.read_csv(data_file, sep="\t")


names = data.columns.to_list()
nodes = data.shape[1]
graph = np.ones((nodes, nodes))


ind = index.Index(args.pert)

def gold_to_matrix(gold, names, size):
    graph = np.zeros((size, size))
    for row in gold.iterrows():
        x = names.index(row[1][0])
        y = names.index(row[1][1])
        z = row[1][2]
        graph[y][x] = z

    return graph

#
def cond_indep(x, y, z, data):
    s_i = ind.single_index(z)
    #su_i = ind.single_up(z)
    #sd_i = ind.single_down(z)
    d_i = ind.dual_index(z, y)
    pd_i = ind.dual_down(z, y)
    pu_i = ind.dual_up(z, y)
    #TODO: Clean up this line
    x_name = names[x]

    junk1 = abs(data.iloc[s_i][x_name] - data.iloc[d_i][x_name]) < 0.01
    #junk1 = (data.iloc[s_i][x_name] == data.iloc[d_i][x_name])
    #junk2 = data.iloc[s_i][x_name] - 1 < 0.05
    if args.p:
        junk2 = abs(data.iloc[s_i][x_name] - data.iloc[pd_i][x_name]) < 0.01
        junk3 = abs(data.iloc[s_i][x_name] - data.iloc[pu_i][x_name]) < 0.01
    else:
        junk2 = True
        junk3 = True

    if junk1 and junk2 and junk3:
        if(args.v):
            print("Gene {} indep of gene {} given gene {}".format(names[x],
                names[y], names[z]))
            print("Gene {} = {} given Gene {} = {} and Gene {} = {}".format(names[x], data.iloc[s_i][names[x]], names[y],
                        data.iloc[s_i][names[y]], names[z], data.iloc[s_i][names[z]]))
            print("Gene {} = {} given Gene {} = {} and Gene {} = {}".format(names[x], data.iloc[d_i][names[x]], names[y],
                        data.iloc[d_i][names[y]], names[z], data.iloc[d_i][names[z]]))

            #print(data.iloc[s_i])
            #print(data.iloc[d_i])
        return True

    else:
        return False


def indep(x, y, data):
    index = ind.single_index(y)
    #This requires steady state info to be in the last row of the matrix
    steady = data.shape[0] - 1
    #TODO: Clean up this line
    x_name = names[x]

    if abs(data.iloc[index][x_name] - float(data.iloc[steady][x_name])) < 0.001:
        if args.v:
            print("{} independent of {}".format(names[x], names[y]))
        return True

    else:
        return False


#print(cond_indep(0, 1, 2, data))
#print(cond_indep(0,2,2,data))
#print(cond_indep(4, 5, 3, data))
#print(indep(3, 5, data))

for i in range(nodes):
    for j in range(nodes):
        graph[i, j] = 1 - indep(i, j, data)


if(args.dual):
    for i in range(nodes):
        for j in range(nodes):
            if graph[i, j] == 1 and i != j:
                for z in range(nodes):
                    if i != z and j != z:
                        if cond_indep(i, j, z, data) and not cond_indep(i, z, j, data):
                            graph[i, j] = 0
                            break

np.fill_diagonal(graph, 0)

gold = gold_to_matrix(gold, names, nodes)


gold_nx = nx.from_numpy_array(gold, create_using=nx.DiGraph)
graph_nx = nx.from_numpy_array(graph, create_using=nx.DiGraph)

score = cdt.metrics.precision_recall(gold_nx, graph_nx)[0]
print(score)
