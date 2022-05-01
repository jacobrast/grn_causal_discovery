import torch 
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import euclidean_distances
from torchinfo import summary
import math
import numpy as np 
import pdb

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('tanh'))
        torch.nn.init.zeros_(m.bias)

def binarify(Adj, threshold, dtype):
    bw = (Adj.clone().detach() >= threshold).float().type(dtype)
    return bw

def get_triu_vec(A):
    row_idx, col_idx = torch.triu_indices(A.shape[0], A.shape[1], offset = 1)
    return A[row_idx, col_idx]

#%%
class dnn_model(torch.nn.Module): # a simple DNN 
    def __init__(self, T, O, H, USE_CUDA=False): # initializing all the weights here
        super(dnn_model, self).__init__() # initializing the nn.module

        self.USE_CUDA = USE_CUDA

        if USE_CUDA == False:
            self.dtype = torch.FloatTensor
        else: # shift to GPU
#            print('shifting to cuda')
            self.dtype = torch.cuda.FloatTensor

        self.O, self.T, self.H = O, T, H

        # **** DNN parameters ****************
        self.DNN = self.getDNN()

    def getDNN(self):
        l1 = nn.Linear(self.T, self.H).type(self.dtype)
#        l2 = nn.Linear(self.H, self.H).type(self.dtype)
        l3 = nn.Linear(self.H, self.O).type(self.dtype)
        return nn.Sequential(l1, nn.ReLU(), #nn.Tanh(), 
#                             l2, nn.ReLU(), #nn.Tanh(),
                             l3, #nn.Tanh(),
                            ).type(self.dtype)

#%%
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        if bias is True:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features))
            nn.init.ones_(self.bias)
        else:
            self.bias = None

    def forward(self, node_feature, adj):
        intermediate_H = torch.matmul(node_feature, self.weight)
        #output = torch.bmm(adj, intermediate_H)
        output = torch.matmul(adj, intermediate_H)
        if self.bias is not None:
            return output + self.bias
        return output

#%%
class GraphEnc(nn.Module):
    def __init__(self, n_inNodeFeat, n_hid, n_outGraphFeat):
        super(GraphEnc, self).__init__()
        self.conv1 = GraphConvLayer(in_features=n_inNodeFeat, out_features=n_hid, bias=True)
        self.conv2 = GraphConvLayer(in_features=n_hid, out_features=int(n_hid * 2), bias=True)
        self.fc1 = nn.Linear(int(n_hid * 2), n_hid, bias=True)
        self.fc2 = nn.Linear(n_hid, n_outGraphFeat, bias=True)
        self.dtype = torch.cuda.FloatTensor

    def forward(self, adj):
        degree = torch.sum(adj, dim=0).unsqueeze_(-1).type(self.dtype)

        # 2-layer GCN
        h = F.relu(self.conv1(degree, adj))
        h2 = F.relu(self.conv2(h, adj.T))

        # readout mean of all node embeddings
        hg = torch.sum(h2, dim=0)
        hg = F.relu(self.fc1(hg.unsqueeze(0)))
        hg = self.fc2(hg)

        return hg


class TopoDiffVAE(nn.Module):
    def __init__(self, graph_size, n_hid, n_latent, n_nodeFeat, n_graphFeat, TF):
        super(TopoDiffVAE, self).__init__()

        self.enc = GraphEnc(n_nodeFeat, n_hid, n_graphFeat)

        self.f_mean = nn.Sequential(
            nn.Linear(n_graphFeat, n_latent)
        )
        self.f_mean.apply(init_weights)

        self.f_var = nn.Sequential(
            nn.Linear(n_graphFeat, n_latent)
        )
        self.f_var.apply(init_weights)


        self.dec = nn.Sequential(
            #nn.Linear(int(graph_size * (graph_size) / 2) + n_latent, int(graph_size * (graph_size - 1)*2/3)),
            nn.Linear(TF * graph_size + n_latent, int(TF * graph_size * 3/2)),
            nn.Tanh(),
            nn.Linear(int(TF * graph_size * 3/2), TF * graph_size)
        )
        self.dec.apply(init_weights)

        self.n_latent = n_latent

    def resample(self, z_vecs, f_mean, f_var):

        z_mean = f_mean(z_vecs)
        z_log_var = -torch.abs(f_var(z_vecs))

        kl_loss = -0.5 * torch.mean(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var))

        epsilon = torch.randn_like(z_mean).type(self.enc.dtype)  # N(0,1) in [batch_size, n_latent]
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon  # latent variable in [batch_size, n_latent]

        return z_vecs, kl_loss

    def latent_noise(self, x_emb, x_batch):

        batch_size = x_emb.size(0)

        diff_noise = torch.randn(batch_size, self.n_latent).type(self.device.dtype)
        latent = torch.cat([x_batch, diff_noise], dim=-1)

        return latent

    def latent_diff(self, x_emb, y_emb, x_batch):
        diff = y_emb - x_emb

        topo_diff_latent, kl = self.resample(diff, self.f_mean, self.f_var)

        latent = torch.cat([x_batch, topo_diff_latent], dim=-1)

        return latent, kl

    def forward(self, x_topo, y_topo, threshold=1e-04):

        binary_x = binarify(x_topo, threshold, self.enc.dtype)
        binary_y = binarify(y_topo, threshold, self.enc.dtype)

        # encoding:
        x_emb = self.enc(binary_x)
        y_emb = self.enc(binary_y)

        # latent:
        x_vec = torch.flatten(x_topo)
        latent, kl = self.latent_diff(x_emb, y_emb, x_vec.unsqueeze(0))

        # decoding:
        y_pred = self.dec(latent)

        # loss:
        y_vec = torch.flatten(y_topo)
        recons_loss = torch.sum((y_pred - y_vec) ** 2, dim=-1).mean()

        kl_hyper = recons_loss / kl
        loss = recons_loss + kl_hyper * kl

        Z = torch.reshape(y_pred.squeeze(0), (x_topo.shape))

        return Z, loss, kl, latent

    def refine(self, x_topo, threshold=1e-04):
        """
        validation
        """
        x_binary = binarify(x_topo, threshold, self.enc.dtype)

        x_emb = self.enc(x_binary)
        latent = self.latent_noise(x_emb, x_batch)
        y_pred = self.dec(latent)

        return y_pred, latent

#%%
class glad_model(torch.nn.Module): # entrywise thresholding
    def __init__(self, L, theta_init_offset, nF, H, graph_size, len_TF, n_hid = 32,
                 n_nodeFeat = 1, n_latent = 16, n_graphFeat = 16, USE_CUDA=False): 
        super(glad_model, self).__init__() # initializing the nn.module
        self.USE_CUDA = USE_CUDA
        if USE_CUDA == False:
            self.dtype = torch.FloatTensor
        else: # shift to GPU
            print('shifting to cuda')
            self.dtype = torch.cuda.FloatTensor
        self.L = L # number of unrolled iterations
        self.theta_init_offset = nn.Parameter(torch.Tensor([theta_init_offset]).type(self.dtype))
        self.nF = nF # number of input features 
        self.H = H # hidden layer size
        self.rho_l1 = self.rhoNN()#nn.Sequential(l1, nn.ReLU(), l2, nn.ReLU()).cuda() # NOTE: just testing
        print('CHECK RHO and theta INITIAL: ', self.rho_l1[0].weight, self.theta_init_offset)
        self.lambda_f = self.lambdaNN()
        self.zero = torch.Tensor([0]).type(self.dtype)

        self.vae = TopoDiffVAE(graph_size, n_hid, n_latent, n_nodeFeat, n_graphFeat, len_TF)

    def rhoNN(self):# per iteration NN
        l1 = nn.Linear(self.nF, self.H).type(self.dtype)
        lH1 = nn.Linear(self.H, self.H).type(self.dtype)
        l2 = nn.Linear(self.H, 1).type(self.dtype)
        return nn.Sequential(l1, nn.Tanh(), 
                             lH1, nn.Tanh(),
                              l2, nn.Sigmoid()).type(self.dtype)

    def lambdaNN(self):
        l1 = nn.Linear(2, self.H).type(self.dtype)
        l2 = nn.Linear(self.H, 1).type(self.dtype)
        return nn.Sequential(l1, nn.Tanh(), 
                              l2, nn.Sigmoid()).type(self.dtype)        

    def eta_forward(self, X, S, k, F3=[]):# step_size):#=1):
        shape1, shape2 = X.shape
        Xr = X.reshape(-1, 1)  
        Sr = S.reshape(-1, 1)
        feature_vector = torch.cat((Xr, Sr), -1)

        if len(F3)>0:
            F3r = F3.reshape(-1, 1)
            feature_vector = torch.cat((feature_vector, F3r), -1)

        rho_val = self.rho_l1(feature_vector).reshape(X.shape) # elementwise thresholding done

        if k < self.L:
            return torch.sign(X)*torch.max(self.zero, torch.abs(X)-rho_val)
        else:
            p1, vae_loss, vae_kl, vae_latent = self.vae.forward(rho_val, X, kl_hyper = 0.01)

    def lambda_forward(self, normF, prev_lambda, k=0):
        feature_vector = torch.Tensor([normF, prev_lambda]).type(self.dtype)
        return self.lambda_f(feature_vector)
