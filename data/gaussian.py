import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import xlogy


class Gaussian():
    # Gaussian
    #
    # Generate two samples represent corelated gaussian distribution
    def __init__(self, sample_size=400, rho=0.9, mean=[0, 0]):
        # sample_size is the number of sample representing the distribution
        # Rho: correlation for the gaussian. This is used to generate an covariance matrix diagonal equal to one and anti-diagonal equal to rho. That means, the covariance between first dimension of first variable and the last dimension of second variable is rho, the covariance between second dimension of first variable and the second last dimension of second variable is rho...
        # mean: an array representing the mean of two variables, first half of array representing the mean of first variable, and second half of the array representing the mean of second variable. We assume the dimension of two variables to be equal, thus we assume even size of mean array.
        self.sample_size = sample_size
        self.mean = mean
        self.rho = rho
        # self.cov = (np.identity(len(self.mean))+self.rho*np.identity(len(self.mean))[::-1]).tolist()
        self.cov = np.array([[1, rho], [rho, 1]])

    @property
    def data(self):
        """[summary]
        Returns:
            [np array] -- [N by 2 matrix]
        """
        if len(self.mean)%2 == 1:
            raise ValueError("length of mean array is assummed to be even")
        
        return np.random.multivariate_normal(
            mean=self.mean,
            cov=self.cov,
            size=self.sample_size)

    @property
    def ground_truth(self):
        # since the covariance matrices of each variable are identity matrices, and the two variables are co-varied with same rho dimension-by-dimension. Therefore we can simplify the ground truth to be the product of mutual information of one-dimension variables and number of dimension of each variable
        if len(self.mean)%2 == 1:
            raise ValueError("length of mean array is assummed to be even")
        dim = len(self.mean)//2
        return -0.5*np.log(1-self.rho**2)*dim
    
    def I(self, x,y):
        # cov = np.array(self.rho)
        if len(self.mean)%2 == 1:
            raise ValueError("length of mean array is assummed to be even")
        dim = len(self.mean)//2
        covMat, mu = self.cov, np.array(self.mean)
        def fxy(x,y):
            X = np.array([x, y])
            temp1 = np.matmul(np.matmul(X-mu , np.linalg.inv(covMat)), (X-mu).transpose())
            return np.exp(-.5*temp1) / (((2*np.pi)**(dim))* np.sqrt(np.linalg.det(covMat))) 

        def fx(x):
            if type(x)==np.float64 or type(x)==float:
                return np.exp(-(x-mu[0])**2/(2*covMat[0,0])) / np.sqrt(2*np.pi*covMat[0,0])
            else:
                temp1 = np.matmul(np.matmul(x-mu[0:dim] , np.linalg.inv(covMat[0:dim,0:dim])), (x-mu[0:dim]).transpose())
                return np.exp(-.5*temp1) / (((2*np.pi)**(dim /2))* np.sqrt(np.linalg.det(covMat[0:dim,0:dim])))

        def fy(y):
            if type(y)==np.float64 or type(y)==float:
                return np.exp(-(y-mu[1])**2/(2*covMat[1,1])) / np.sqrt(2*np.pi*covMat[1,1])*dim
            else:
                temp1 = np.matmul(np.matmul(y-mu[-dim:] , np.linalg.inv(covMat[-dim:,-dim:])), (y-mu[-dim:]).transpose())
                return np.exp(-.5*temp1) / (((2*np.pi)**(dim /2))* np.sqrt(np.linalg.det(covMat[-dim:,-dim:])))

        return np.log(fxy(x, y)/(fx(x)*fy(y)))