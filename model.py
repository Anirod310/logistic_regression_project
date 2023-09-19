import numpy as np
import math



def sigmoid(z):
    
    """Convert the original linear function to a sigmoid function, 
    wich is the base function for the logistic regression model.
    
    Args:
        z (float): base linear function of the model : fwb(x) = dot_product(w,x) + b

    Returns:
        float: result of the sigmoid function : fwb(x) = g(dot_product(w,x) + b) 
    """    #The function of the model is a sigmoid function, which fit perfectly with the expected results.
    
    g = 1/(1 + np.exp(-z))
    
    return g



def cost_function(X, y, w, b, lambda_ = 1):
    """Calculates the difference between the results with the parameters 
    of the model and the real result, expressed as a cost.

    Args:
        X (ndarray): training data
        y (ndarray): training results
        w (ndarray): weights
        b (int): bias
        lambda_ (int): regularisation term. Defaults to 1.

    Returns:
        float: value between 0 and 1. The more the value is far to 0, 
            the more the cost is high and the parameters inefficient.
    """    
    m = X.shape[0]
    cost = 0
    reg_cost = sum(np.square(w))
    
    for i in range(m):
        f_wb = sigmoid(np.dot(X[i], w) + b)
        cost += -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)
    
    cost_without_reg = cost / m 
    total_cost = cost_without_reg + (lambda_/(2 * m)) * reg_cost
    
    return total_cost



def gradient_function(X, y, w, b, lambda_ = 1):
    """Computes the gradient of the function with respect to the weights and bias

    Args:
        X (ndarray): training data
        y (_tyndarraype_): training results
        w (ndarray): weights
        b (int): bias
        lambda_ (int): regularisation term. Defaults to 1.

    Returns:
        float, ndarray: sum of the derivatives of the cost function, 
            with respect to the weights and bias
    """    
    m, n = X.shape
    
    db = 0
    dw = np.zeros((n,))
    
    for i in range(m):
        f_wb = sigmoid(np.dot(X[i], w) + b)
        err = f_wb - y[i]
        
        db += err
        for j in range(n):
            dw[j] += err * X[i, j]

    
    db /= m                   
    dw /= m
    
    for j in range(n):
        dw[j] += (lambda_/m) * w[j]
    
    return db, dw


def gradient_descent(X, y, w, b, cost_function, gradient_function, num_iter, alpha, lambda_):
    """Updates the inital w and b paramaters to minimize the cost function, 
        until the cost function converges to a minimum value.
    
    Args:
        X (ndarray): training data
        y (_tyndarraype_): training results
        w (ndarray): weights
        b (int): bias
        compute_cost (float): cost with respect to the parameters w and b 
        compute_gradient (float): derivatives of the cost, with respect to w and b
        num_iter (int): number of iterations
        alpha (float): learning rate
        lambda_ (int): regularisation term. Defaults to 1.

    Returns:
        ndarray, float, list, list: final values of w and b, 
            and a list that contains the cost and the weights (in order to print the learning during the loop).
    """    
    cost_function_total_history = []
    cost_function_for_print_history = []
    w_history = []
    
    
    for i in range(num_iter):
        
        db, dw = gradient_function(X, y, w, b, lambda_)
        
        w = w - alpha * dw
        b = b - alpha * db
        
        if i < 10000:
            cost = cost_function(X, y, w, b, lambda_)
            cost_function_total_history.append(cost)
        
        if i% math.ceil(num_iter/10) == 0 or i == (num_iter-1):
            w_history.append(w)
            cost_function_for_print_history.append(format(float(cost_function_total_history[-1]),".5f"))
    print('\n')
    
    return w, b, cost_function_for_print_history, w_history




