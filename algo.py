import pandas as pd
import numpy as np
import networkx as nx
import cdt
import pdb
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('knockout', type=str)
parser.add_argument('gold', type=str)
parser.add_argument('knockdown', type=str)

args = parser.parse_args()

#data_file = "knockdown.tsv"
data_file = args.knockdown
data = pd.read_csv(data_file, sep="\t")

#mode = data.mode().head(1).values.tolist()[0][1:]
mode = data.mode().head(1)

#data_file = "gold.tsv"
data_file = args.gold
gold = pd.read_csv(data_file, sep="\t", header=None)

#data_file = "knockout.tsv"
data_file = args.knockout
data = pd.read_csv(data_file, sep="\t")


names = data.columns.to_list()
nodes = data.shape[1]
graph = np.ones((nodes, nodes))


def gold_to_matrix(gold, names, size):
    graph = np.zeros((size, size))
    for row in gold.iterrows():
        x = names.index(row[1][0])
        y = names.index(row[1][1])
        z = row[1][2]
        graph[x][y] = z

    return graph


def indep(x, y, row, steady):
    if abs(row[y] - float(steady[y])) > 0.01:
        return False

    return True


for i, row in data.iterrows():
    for j, item in enumerate(row):
        graph[i, j] = 1 - indep(names[i], names[j], row, mode)
                
np.fill_diagonal(graph, 0)

gold = gold_to_matrix(gold, names, nodes)

gold_nx = nx.from_numpy_array(gold, create_using=nx.DiGraph)
graph_nx = nx.from_numpy_array(graph, create_using=nx.DiGraph)

score = cdt.metrics.precision_recall(gold_nx, graph_nx)[0]
print(score)

pdb.set_trace()

#nx.draw_networkx(G, arrows=True)
#plt.show()
