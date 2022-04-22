import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

data_file = "knockdown.tsv"
data = pd.read_csv(data_file, sep="\t")

#mode = data.mode().head(1).values.tolist()[0][1:]
mode = data.mode().head(1)

data_file = "knockout.tsv"
data = pd.read_csv(data_file, sep="\t")


names = data.columns
nodes = data.shape[1]
graph = np.ones((nodes, nodes))

def indep(x, y, row, steady):
    if abs(row[y] - float(steady[y])) > 0.1:
        return False

    return True


for i, row in data.iterrows():
    for j, item in enumerate(row):
        graph[i, j] = 1 - indep(names[i], names[j], row, mode)
                
np.fill_diagonal(graph, 0)
G = nx.from_numpy_array(graph)
nx.draw_networkx(G, arrows=True)
plt.show()
