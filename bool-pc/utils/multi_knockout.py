import numpy as np
import pdb
from math import comb

fname = "stable.tsv"
data = np.loadtxt(fname, dtype='str', delimiter='\t')[1]

l = data.shape[0]
n = l + comb(l, 2) + 2 * (l * (l-1))


#Why add 4?
#1 line for header, requried by GNW
#1 line for dual knockout steady state
#1 line for dual knockdown values
#1 line for dual knockup values
out = np.tile(data, (n+4, 1))

print(out.shape)


#Set first row to junk
acc = 1

#Single knockout
for i in range(l):
    out[acc, i] = 0
    acc += 1

#Dual knockout
for i in range(l):
    for j in range(0, i+1):
        if j == i:
            continue

        out[acc, i] = 0
        out[acc, j] = 0

        acc+=1

#Dual perturbation down
for i in range(l):
    for j in range(l):
        if j == i:
            continue

        out[acc, i] = 0
        out[acc, j] = float(out[acc, j]) - 2*float(out[acc, j])

        acc+=1


#Dual perturbation up
for i in range(l):
    for j in range(l):
        if j == i:
            continue

        out[acc, i] = 0
        out[acc, j] = float(out[acc, j]) + 2*float(out[acc, j])

        acc+=1

out[acc] = out[acc].astype('float32') + 2 * out[acc].astype('float32')
acc += 1
out[acc] = out[acc].astype('float32') - 2 * out[acc].astype('float32')

np.savetxt("knockout_perturbations.tsv", out.astype('half'), delimiter="\t")
