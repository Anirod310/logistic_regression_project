import numpy as np
from model import sigmoid


def prediction_function(X, w, b):
    """Uses the paramaters learned by gradient descent to predict the value of new features

    Args:
        X (ndarray): training data
        w (ndarray): weights
        b (int): bias
    Returns:
        ndarray : array that contains all the predictions for each feature.
    """    
    m = X.shape[0]
    p = np.zeros(m)
    
    for i in range(m):
        f_wb = sigmoid(np.dot(X[i], w) + b)
        p[i] = 1 if f_wb >= 0.5 else 0
    
    return p

def predict_one_data(X, w, b):
    
    f_wb = sigmoid(np.dot(X, w) + b )
    p = 1 if f_wb >= 0.5 else 0
    
    return p 
