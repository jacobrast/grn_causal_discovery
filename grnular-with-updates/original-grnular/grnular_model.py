import torch 
import torch.nn as nn
import pdb

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


class glad_model(torch.nn.Module): # entrywise thresholding
    def __init__(self, L, theta_init_offset, nF, H, USE_CUDA=False): # initializing all the weights here
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
        return torch.sign(X)*torch.max(self.zero, torch.abs(X)-rho_val)

    def lambda_forward(self, normF, prev_lambda, k=0):
        feature_vector = torch.Tensor([normF, prev_lambda]).type(self.dtype)
        return self.lambda_f(feature_vector)
