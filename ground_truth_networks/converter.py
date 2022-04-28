import numpy as np
import pandas as pd
import argparse


#Function 1: Read data from DREAM standard
def read_dream(fname):
    data = pd.read_csv(fname, sep="\t", header=None)
    names = []
    for row in data.iterrows():
        if row[1][0] not in names:
            names.append(row[1][0])

    matrix = np.zeros((len(names), len(names)))
    for row in data.iterrows():
        x = names.index(row[1][0])
        y = names.index(row[1][1])
        z = row[1][2]
        matrix[y][x] = z

    return names, matrix


#Function 2: Read data from BEELINE standard
def read_beeline(fname):
    data = pd.read_csv(fname, sep=",", header=None)
    names = []

    for row in data.iterrows():
        if row[1][0] not in names:
            names.append(row[1][0])

        if row[1][1] not in names:
            names.append(row[1][1])

    matrix = np.zeros((len(names), len(names)))
    for row in data.iterrows():
        x = names.index(row[1][0])
        y = names.index(row[1][1])
        matrix[y][x] = 1

    return names, matrix


#Function 3: Write data to DREAM standard
def write_dream(fname, names, matrix):
    names_out = []
    l = matrix.shape[0]
    out = np.zeros((l*(l-1), 3), dtype="object")
    acc = 0

    for name in names:
        if str.isdigit(str(name)):
            names_out.append("G{}".format(str(name)))
        else:
            names_out.append(name)

    names=names_out

    for i in range(l):
        for j in range(l):
            if i == j:
                continue
            out[acc][1] = names[i]
            out[acc][0] = names[j]
            out[acc][2] = matrix[i][j]
            acc += 1

    np.savetxt(fname, out, delimiter="\t", fmt=["%s", "%s", "%d"])

#Function 4: Write data to BEELINE standard
def write_beeline(fname, names, matrix):
    l = matrix.shape[0]
    out = np.zeros((l*l, 2), dtype="object")
    acc = 0
    for i in range(l):
        for j in range(l):
            if i == j:
                continue
            if matrix[i][j] == 1:
                out[acc][1] = names[i]
                out[acc][0] = names[j]
                acc += 1

    np.savetxt(fname, out[0:acc], delimiter=",", fmt="%s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('input_type', choices=['dream', 'beeline'])
    args = parser.parse_args()

    if args.input_type == 'beeline':
        names, matrix = read_beeline(args.input)
        write_dream(args.output, names, matrix)

    else:
        names, matrix = read_dream(args.input)
        write_beeline(args.output, names, matrix)
