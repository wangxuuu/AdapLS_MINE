from data.mix_gaussian import MixedGaussian
from data.gaussian import Gaussian
import numpy as np
import torch

def generate_data(distribution='Gaussian', sample_size=100, rho=0.9, d=2):
    # np.random.seed(seed)
    # # initialize random seed
    # torch.manual_seed(seed)
    mu1 = 0
    mu2 = 0

    X = np.zeros((sample_size,d))
    Y = np.zeros((sample_size,d))
    # print('start generating')
    if distribution=='Gaussian':
        mg = Gaussian(sample_size=sample_size,rho=rho)
    else:
        mg = MixedGaussian(sample_size=sample_size,mean1=mu1, mean2=mu2,rho1=rho,rho2=-rho)
    # print('end generate')
    mi = mg.ground_truth * d
    data = mg.data
    for j in range(d):
        data = mg.data
        X[:,j] = data[:,0]
        Y[:,j] = data[:,1]
    X = torch.Tensor(X)
    Y = torch.Tensor(Y)
    XY = torch.cat((X, Y), dim=1)
    return XY, X, Y, mi