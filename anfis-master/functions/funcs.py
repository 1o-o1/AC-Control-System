import numpy as np
from scipy.special import expit

def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    # pos_mask = (x >= 0)
    # neg_mask = (x < 0)
    # z = np.zeros_like(x)
    # z[pos_mask] = np.exp(-x[pos_mask])
    # z[neg_mask] = np.exp(x[neg_mask])
    # top = np.ones_like(x)
    # top[neg_mask] = z[neg_mask]
    x = np.asarray(x,dtype=np.longdouble)
    for i in range(5):
        x[i] = min(x[i],100)
        x[i] = max(x[i],-100)
    return 1 / (1 + np.exp(x))
     

def lab_func(ai, bi, x):
    mu = 1 / (1 + np.exp(bi.T * (x - ai.T)))
    # mu = 1 / (1 + np.exp(bi*(x - ai)))
    # return mu

