import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

"""
modify from minee.py to minee_multitask.py
1. the part between two lines of #### is the codes added
2. the line following #### is the code commented out
3. the line between  3*" ### and 3*" are the codes commented out
4. the arg share_size can be processed better
"""

def _resample(data, batch_size, replace=False):
    # Resample the given data sample.
    index = np.random.choice(
        range(data.shape[0]), size=batch_size, replace=replace)
    batch = data[index]
    return batch


def _uniform_sample(data, batch_size):
    # Sample the reference uniform distribution
    data_min = data.min(dim=0)[0]
    data_max = data.max(dim=0)[0]
    return (data_max - data_min) * torch.rand((batch_size, data_min.shape[0])) + data_min

"""####
def _div(net, data, ref):
    # Calculate the divergence estimate using a neural network
    mean_f = net(data).mean()
    log_mean_ef_ref = torch.logsumexp(net(ref), 0) - np.log(ref.shape[0])
    return mean_f - log_mean_ef_ref
"""

####
def _div(net, data, ref, net_share = None):
    # Calculate the divergence estimate using a neural network
    # in MINEE().forward(), X_share and Y_share are calculated as intermediate result for saving computation,hence we do not need net_share.
    if net_share == None:
        mean_f = net(data).mean()
        log_mean_ef_ref = torch.logsumexp(net(ref), 0) - np.log(ref.shape[0])
        
    elif len(net_share) == 1:
        mean_f = net(net_share[0](data)).mean()
        log_mean_ef_ref = torch.logsumexp(net(
                               net_share[0](ref)), 0) - np.log(ref.shape[0])
    elif len(net_share) == 2:
        mean_f = net(torch.cat((net_share[0](data[:,0:data.shape[1]//2]), 
                          net_share[1](data[:,data.shape[1]//2:data.shape[1]+1])), dim = 1)).mean()
        log_mean_ef_ref = torch.logsumexp(net(
                torch.cat((net_share[0](ref[:,0:ref.shape[1]//2]), 
                          net_share[1](ref[:,ref.shape[1]//2:ref.shape[1]+1])), dim = 1)), 0
                                         ) - np.log(ref.shape[0])
    return mean_f - log_mean_ef_ref
    ####

class MINEE():
    r"""Class for Mutual Information Neural Entropic Estimation. 

    The mutual information is estimated using neural estimation of divergences 
    to uniform reference distribution.

    Arguments:
    X (tensor): samples of X
        dim 0: different samples
        dim 1: different components
    Y (tensor): samples of Y
        dim 0: different samples
        dim 1: different components
    ref_batch_factor (float, optional): multiplicative factor to increase 
        reference sample size relative to sample size
    lr (float, optional): learning rate
    hidden_size (int, optional): size of the hidden layers
    """
    class Net(nn.Module):
        # Inner class that defines the neural network architecture
        ####def __init__(self, input_size=2, hidden_size=100, sigma=0.02):
        ####s 
        def __init__(self, input_size=2, hidden_size=100, sigma=0.02, share_size = 4):
        ####    
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
            output = self.fc3(output)
            return output
    ####    
    class ShareNet(nn.Module):
        # Inner class that defines the neural network architecture
        def __init__(self, input_size=2, hidden_size=100, sigma=0.02, share_size = 4):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, share_size)
            nn.init.normal_(self.fc1.weight, std=sigma)
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.normal_(self.fc2.weight, std=sigma)
            nn.init.constant_(self.fc2.bias, 0)
            nn.init.normal_(self.fc3.weight, std=sigma)
            nn.init.constant_(self.fc3.bias, 0)

        def forward(self, input):
            output = F.elu(self.fc1(input))
            output = F.elu(self.fc2(output))
            output = self.fc3(output)
            return output        
    ####

    ####def __init__(self, X, Y, batch_size=32, ref_batch_factor=1, lr=1e-3, hidden_size=100):
    
    #### share_size is the dimension for the intermediate shared feature (currently the X and Y for Net()) 
    def __init__(self, X, Y, batch_size=32, ref_batch_factor=1, lr=1e-3, hidden_size=100, share_size = 4):
    ####
        self.lr = lr
        self.batch_size = batch_size
        self.ref_batch_factor = ref_batch_factor
        self.X = X
        self.Y = Y
        self.XY = torch.cat((self.X, self.Y), dim=1)
        
        ####
        self.share_size = share_size # can also consider to set share_size to twice X.shape[1] by default
        ####
        
        #### X_share_net and Y_share_net borrow Net() here, also can define and use a new NN structure
        self.X_share_net = MINEE.ShareNet(input_size=X.shape[1], hidden_size=100, share_size = self.share_size)
        self.Y_share_net = MINEE.ShareNet(input_size=Y.shape[1], hidden_size=100, share_size = self.share_size)
        self.X_share_optimizer = optim.Adam(self.X_share_net.parameters(), lr=lr)
        self.Y_share_optimizer = optim.Adam(self.Y_share_net.parameters(), lr=lr)
        # here only use the dimension of X_share and Y_share
        self.X_share = torch.zeros(X.shape[0], self.share_size)  
        self.Y_share = torch.zeros(Y.shape[0], self.share_size)
        ####
        
        ####
        #we did not store self.XY_share
        #self.XY_share = torch.cat((self.X_share, self.Y_share), dim=1)
        self.X_ref = _uniform_sample(X, batch_size=int(
            self.ref_batch_factor * X.shape[0]))
        self.Y_ref = _uniform_sample(Y, batch_size=int(
            self.ref_batch_factor * Y.shape[0]))

        self.XY_net = MINEE.Net(
            input_size=self.X_share.shape[1]+self.Y_share.shape[1], hidden_size=100)
        self.X_net = MINEE.Net(input_size=self.X_share.shape[1], hidden_size=100)
        self.Y_net = MINEE.Net(input_size=self.Y_share.shape[1], hidden_size=100)
        self.XY_optimizer = optim.Adam(self.XY_net.parameters(), lr=lr)
        self.X_optimizer = optim.Adam(self.X_net.parameters(), lr=lr)
        self.Y_optimizer = optim.Adam(self.Y_net.parameters(), lr=lr)
        ####

        """ ####This part is the previous code        
        self.X_ref = _uniform_sample(X, batch_size=int(
            self.ref_batch_factor * X.shape[0]))
        self.Y_ref = _uniform_sample(Y, batch_size=int(
            self.ref_batch_factor * Y.shape[0]))

        self.XY_net = MINEE.Net(
            input_size=X.shape[1]+Y.shape[1], hidden_size=100)
        self.X_net = MINEE.Net(input_size=X.shape[1], hidden_size=100)
        self.Y_net = MINEE.Net(input_size=Y.shape[1], hidden_size=100)
        self.XY_optimizer = optim.Adam(self.XY_net.parameters(), lr=lr)
        self.X_optimizer = optim.Adam(self.X_net.parameters(), lr=lr)
        self.Y_optimizer = optim.Adam(self.Y_net.parameters(), lr=lr)
        """
    def step(self, iter=1):
        r"""Train the neural networks for one or more steps.

        Argument:
        iter (int, optional): number of steps to train.
        """
        for i in range(iter):
            self.XY_optimizer.zero_grad()
            self.X_optimizer.zero_grad()
            self.Y_optimizer.zero_grad()
            
            ####
            self.X_share_optimizer.zero_grad()
            self.Y_share_optimizer.zero_grad()
            ####
            
            batch_XY = _resample(self.XY, batch_size=self.batch_size)
            batch_X = _resample(self.X, batch_size=self.batch_size)
            batch_Y = _resample(self.Y, batch_size=self.batch_size)
            batch_X_ref = _uniform_sample(self.X, batch_size=int(
                self.ref_batch_factor * self.batch_size))
            batch_Y_ref = _uniform_sample(self.Y, batch_size=int(
                self.ref_batch_factor * self.batch_size))
            batch_XY_ref = torch.cat((batch_X_ref, batch_Y_ref), dim=1)

            ####batch_loss_XY = -_div(self.XY_net, batch_XY, batch_XY_ref)
            ####batch_loss_XY.backward()
            ####self.XY_optimizer.step()

            ####batch_loss_X = -_div(self.X_net, batch_X, batch_X_ref)
            ####batch_loss_X.backward()
            ####self.X_optimizer.step()

            ####batch_loss_Y = -_div(self.Y_net, batch_Y, batch_Y_ref)
            ####batch_loss_Y.backward()
            ####self.Y_optimizer.step()

            #### loss from X_net and Y_net will also update X_share_net and Y_share_net
            batch_loss_XY = -_div(self.XY_net, batch_XY, batch_XY_ref, 
                                  net_share = [self.X_share_net, self.Y_share_net])
            batch_loss_X = -_div(self.X_net, batch_X, batch_X_ref, net_share = [self.X_share_net])
            batch_loss_Y = -_div(self.Y_net, batch_Y, batch_Y_ref, net_share = [self.Y_share_net])
            batch_loss = batch_loss_XY + batch_loss_X + batch_loss_Y
            batch_loss.backward()
            self.XY_optimizer.step()
            self.X_optimizer.step()
            self.Y_optimizer.step()
            self.X_share_optimizer.step()
            self.Y_share_optimizer.step()
            ####

    def forward(self, X=None, Y=None):
        r"""Evaluate the neural networks to return an array of 3 divergences estimates 
        (dXY, dX, dY).

        Outputs:
            dXY: divergence of sample joint distribution of (X,Y) 
                to the uniform reference
            dX: divergence of sample marginal distribution of X 
                to the uniform reference
            dY: divergence of sample marginal distribution of Y
                to the uniform reference

        Arguments:
            X (tensor, optional): samples of X.
            Y (tensor, optional): samples of Y.
        By default, X and Y for training is used. 
        The arguments are useful for testing/validation with a separate data set.
        """
        
        """ ####This part is the previous code
        XY = None
        if X is None or Y is None:
            XY, X, Y = self.XY, self.X, self.Y
        else:
            XY = torch.cat((X, Y), dim=1)
            
        X_ref = _uniform_sample(X, batch_size=int(
            self.ref_batch_factor * X.shape[0]))
        Y_ref = _uniform_sample(Y, batch_size=int(
            self.ref_batch_factor * Y.shape[0]))
        XY_ref = torch.cat((X_ref, Y_ref), dim=1)
        
        dXY = _div(self.XY_net, XY, XY_ref).cpu().item()
        dX = _div(self.X_net, X, X_ref).cpu().item()
        dY = _div(self.Y_net, Y, Y_ref).cpu().item()            
        """
        
        #### the calculation of X_share and Y_share saves computation
        XY_share = None
        if X is None or Y is None:
            X, Y = self.X, self.Y
        else:
            pass

        X_share = self.X_share_net(X)
        Y_share = self.Y_share_net(Y)
        XY_share = torch.cat((X_share, Y_share), dim=1)

        X_ref = _uniform_sample(X, batch_size=int(
            self.ref_batch_factor * X.shape[0]))
        Y_ref = _uniform_sample(Y, batch_size=int(
            self.ref_batch_factor * Y.shape[0]))
        X_ref_share = self.X_share_net(X_ref)
        Y_ref_share = self.Y_share_net(Y_ref)
        XY_ref_share = torch.cat((X_ref_share, Y_ref_share), dim=1)
                
        dXY = _div(self.XY_net, XY_share, XY_ref_share).cpu().item()
        dX = _div(self.X_net, X_share, X_ref_share).cpu().item()
        dY = _div(self.Y_net, Y_share, Y_ref_share).cpu().item()
        return dXY, dX, dY
        ####

    def estimate(self, X=None, Y=None):
        r"""Return the mutual information estimate.

        Arguments:
            X (tensor, optional): samples of X.
            Y (tensor, optional): samples of Y.
        By default, X and Y for training is used. 
        The arguments are useful for testing/validation with a separate data set.
        """
        dXY, dX, dY = self.forward(X, Y)
        return dXY - dX - dY

    def state_dict(self):
        r"""Return a dictionary storing the state of the estimator.
        """
        return {
            ####
            'X_share_net': self.X_share_net.state_dict(),
            'X_share_optimizer': self.X_share_optimizer.state_dict(),
            'Y_share_net': self.Y_share_net.state_dict(),
            'Y_share_optimizer': self.Y_share_optimizer.state_dict(),
            'X_share': self.X_share, # store this can avoid use forward() to obtain this in initiallization, if do not want to store this, the __init__() need change regarding this 
            'Y_share': self.Y_share,
            ####
            'XY_net': self.XY_net.state_dict(),
            'XY_optimizer': self.XY_optimizer.state_dict(),
            'X_net': self.X_net.state_dict(),
            'X_optimizer': self.X_optimizer.state_dict(),
            'Y_net': self.Y_net.state_dict(),
            'Y_optimizer': self.Y_optimizer.state_dict(),
            'X': self.X,
            'Y': self.Y,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'ref_batch_factor': self.ref_batch_factor
        }

    def load_state_dict(self, state_dict):
        r"""Load the dictionary of state state_dict.
        """
        ####
        self.X_share_net.load_state_dict(state_dict['X_share_net'])
        self.X_share_optimizer.load_state_dict(state_dict['X_share_optimizer'])
        self.Y_share_net.load_state_dict(state_dict['Y_share_net'])
        self.Y_share_optimizer.load_state_dict(state_dict['Y_share_optimizer'])
        self.X_share = state_dict['X_share']
        self.Y_share = state_dict['Y_share']
        ####
        self.XY_net.load_state_dict(state_dict['XY_net'])
        self.XY_optimizer.load_state_dict(state_dict['XY_optimizer'])
        self.X_net.load_state_dict(state_dict['X_net'])
        self.X_optimizer.load_state_dict(state_dict['X_optimizer'])
        self.Y_net.load_state_dict(state_dict['Y_net'])
        self.Y_optimizer.load_state_dict(state_dict['Y_optimizer'])
        self.X = state_dict['X']
        self.Y = state_dict['Y']
        if 'lr' in state_dict:
            self.lr = state_dict['lr']
        if 'batch_size' in state_dict:
            self.batch_size = state_dict['batch_size']
        if 'ref_batch_factor' in state_dict:
            self.ref_batch_factor = state_dict['ref_batch_factor']
