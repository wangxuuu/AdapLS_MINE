import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import math

def resample(data, batch_size, replace=False):
    # Resample the given data sample.
    index = np.random.choice(
        range(data.shape[0]), size=batch_size, replace=replace)
    batch = data[index]
    return batch

def uniform_sample(data, batch_size):
    # Sample the reference uniform distribution
    data_min = data.min(dim=0)[0]
    data_max = data.max(dim=0)[0]
    return (data_max - data_min) * torch.rand((batch_size, data_min.shape[0])) + data_min

def div(net, data, ref):
    # Calculate the divergence estimate using a neural network
    mean_f = net(data).mean()
    log_mean_ef_ref = torch.logsumexp(net(ref), 0) - np.log(ref.shape[0])
    return mean_f - log_mean_ef_ref

def mi_estimate(model, test_XY, gamma, alpha):
    # clip the output of neural networks
    pre = model(test_XY).clamp(min=gamma, max=1-gamma)
    MI_est = torch.log(alpha*pre/((1-pre).clamp(min=gamma, max=1-gamma))).mean()
    # pre = model(test_XY)
    # MI_est = torch.log(alpha*pre/((1-pre))).mean()

    return MI_est

def js_mi_estimate(model, test_XY, gamma):
    pre = model(test_XY).clamp(min=gamma, max=1-gamma)
    MI_est = torch.log(pre/((1-pre).clamp(min=gamma, max=1-gamma))).mean()
    return MI_est

def mi_estimate_KL(model, xy, ref_xy, gamma):
    num_reference = ref_xy.shape[0]
    num_data = xy.shape[0]
    alpha = num_reference/num_data
    pre_xy = model(xy).clamp(min=gamma, max=(1-gamma))
    pre_xy_ref = model(ref_xy).clamp(min=gamma, max=(1-gamma))
    
    mean_f = torch.log((pre_xy/(1-pre_xy))*alpha).mean()
    log_mean_ef_ref = torch.log((pre_xy_ref/(1-pre_xy_ref)*alpha).mean())

    return (mean_f - log_mean_ef_ref).item()

class Net(nn.Module):
    # Inner class that defines the neural network architecture
    def __init__(self, input_size=2, hidden_size=100, sigma=0.02):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight, std=sigma)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=sigma)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=sigma)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = torch.sigmoid(self.fc3(output))
        return output

class Feature_map(nn.Module):
    # compress the features from the original sample space
    def __init__(self, input_size=2, hidden_size=100, output_size=2, sigma=0.02):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        # self.l1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(True))
        # self.l2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(True))
        self.l3 = nn.Linear(hidden_size, output_size)
        nn.init.normal_(self.l1.weight, std=sigma)
        nn.init.constant_(self.l1.bias, 0)
        nn.init.normal_(self.l2.weight, std=sigma)
        nn.init.constant_(self.l2.bias, 0)
        nn.init.normal_(self.l3.weight, std=sigma)
        nn.init.constant_(self.l3.bias, 0)

    def forward(self, input):
        out = F.elu(self.l1(input))
        out = F.elu(self.l2(out))
        # out = self.l1(input)
        # out = self.l2(out)
        out = self.l3(out)
        return out


class CorrelatedStandardNormals(object):
    def __init__(self, dim, rho, device):
        assert abs(rho) <= 1
        self.dim = dim
        self.rho = rho
        self.pdf = MultivariateNormal(torch.zeros(dim).to(device),
                                      torch.eye(dim).to(device))

    def I(self):
        num_nats = - self.dim / 2 * math.log(1 - math.pow(self.rho, 2)) \
                   if abs(self.rho) != 1.0 else float('inf')
        return num_nats

    def hY(self):
        return 0.5 * self.dim * math.log(2 * math.pi)

    def draw_samples(self, num_samples):
        X, ep = torch.split(self.pdf.sample((2 * num_samples,)), num_samples)
        Y = self.rho * X + math.sqrt(1 - math.pow(self.rho, 2)) * ep
        return X, Y

