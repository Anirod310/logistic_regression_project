import numpy as np
from read_data import usable_data
from model import gradient_descent, cost_function, gradient_function
from predict import prediction_function, predict_one_data


X_train, X_test, y_train, y_test = usable_data('./data.csv')


#all inital values 
#(PS : The learning rate and the number of iterations were chosen to product the best accuracy (78.3854) )
w = np.zeros(X_train.shape[1])
b = 0
iterations = 500
alpha = 0.1


w,b, cost_function_for_print_history,_ = gradient_descent(X_train ,y_train, 
                                                        w, b, cost_function, gradient_function, iterations, alpha, 0)

#prediction function on one data, change 'X_train[1]' by any np array of dim(1, 8) to use it : 
prediction_of_one_result = predict_one_data(X_test[1], w, b)

#prediction function on all the dataset, usefull for the accuracy prediction : 
prediction_on_all_data = prediction_function(X_test, w, b)


#print the dataset :
print(f'Dataset : {X_train}\n\nDimensions of the dataset : {X_train.shape}\n')

#print the values of w and b :
print(f'Values of w : {w}\n\nValue of b : {b}\n')  

#print the cost, normally decrasing during the gradient descent process : 
print(f'Cost each (iterations*0.1) : {cost_function_for_print_history}\n')  

#print the accuracy of the model, based on the same training set : 
print(f'The model is accurate at {np.mean(prediction_on_all_data == y_test)*100} %\n')

#print the prediction on one single data : 
print(f'Predict function test : {prediction_of_one_result}\n')
