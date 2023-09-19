import csv 
import numpy as np



def z_normalisation(X):
    """Takes the training data and normalize them with the z_normalisation formula.

    Args:
        X (ndarray): training data from the original file

    Returns:
        ndarray: training data normalized, easier to use because same order of magnitude
    """   
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    
    X_norm = (X - mu) / sigma
    
    return X_norm


def usable_data(filename):#The data are in a csv file, so the first part is to transform it in data usable in code
    with open(filename, 'r') as csvfile :
        reader = csv.reader(csvfile)
        
        data = []
        for row in reader:
            data.append(row)
        
        data_np = np.array(data[1:], dtype='float')#All the data except the first line, which corresponds to the name of the features
        
    X_train = z_normalisation(data_np[:615, :-1])#80% of the dataset is used as training_set, for the features x and the outcome y
    X_test = z_normalisation(data_np[615:, :-1])#20% of the dataset is used as testing_set, for the features x and the outcome y
    y_train = data_np[:615, -1]
    y_test = data_np[615:, -1]
    
    return X_train, X_test, y_train, y_test