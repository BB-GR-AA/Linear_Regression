import torch

def linear_model(X,w,b):
    ''' Y = Xw + b'''
    return torch.add(torch.matmul(X,w),b)

def SSE(Y_hat, Y_obs):
    ''' Sum of squared errors on Y_obs after the fit.'''
    return ((Y_hat - Y_obs)**2).sum() 
