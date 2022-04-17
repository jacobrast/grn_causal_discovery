import pandas as pd
import numpy as np
import pdb

data_file = "knockdown.tsv"
#data_file = "knockout.tsv"
data = pd.read_csv(data_file, sep="\t")

names = data.columns
nodes = data.shape[1] - 1
graph = np.ones((nodes, nodes))
t_steps = 21


def indep(x, y, exp, data):
    mode_y = float(data[y].mode())

    for index, row in exp.iterrows():
        if abs(row[y] - mode_y) > 0.1:
            return False

    return True
                

j = 0
while ((j+1)*t_steps <= len(data)):
    exp = data.iloc[j*t_steps:(j+1)*t_steps]
    pert = j+1

    for i in range(1, nodes+1):
        graph[pert-1, i-1]=1-indep(names[pert], names[i], exp, data)

    j = j + 1

print(graph)
