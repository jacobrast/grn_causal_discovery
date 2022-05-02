import pandas as pd
import pdb

class Index():
    def __init__(self, file_name):
        self.data = pd.read_csv(file_name, sep="\t")
        l = self.data.shape[1]
        cols = []
        for i in range(l):
            cols.append(i)

        self.data.columns = cols


    def single_index(self, x):
        return self.data.loc[self.data[x] == 0].head(1).index[0]


    def dual_index(self, x, y):
        return self.data.loc[(self.data[x] == 0) & (self.data[y] == 0)].head(1).index[0]

    '''
    def single_down(self, x):
        return self.data.loc[(self.data[x] == self.data.iloc[-2][x])].head(1).index[0]

    def single_up(self, x):
        return self.data.loc[(self.data[x] == self.data.iloc[-3][x])].head(1).index[0]
    '''

    def dual_down(self, x, y):
        return self.data.loc[(self.data[x] == 0) & (self.data[y] == self.data.iloc[-2][y])].head(1).index[0]

    def dual_up(self, x, y):
        return self.data.loc[(self.data[x] == 0) & (self.data[y] == self.data.iloc[-3][y])].head(1).index[0]


'''
    def dual_down(self, x, y):
        return self.data.loc[(self.data[x] == self.data.iloc[-2][x]) & (self.data[y] == self.data.iloc[-2][y])].head(1).index[0]

    def dual_up(self, x, y):
        return self.data.loc[(self.data[x] == self.data.iloc[-3][x]) & (self.data[y] == self.data.iloc[-3][y])].head(1).index[0]
'''
