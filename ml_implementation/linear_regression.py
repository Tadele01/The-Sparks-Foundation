'''
To build  a linear regression model
---> model(mathematical model)
---> cost function
---> cost minimizer

model = linear model (our output is a linear combination of its input)
       eg. w0 + w1*x1 + w2*x2 + ....wn*xn

cost function = mean squared error
       1/m*sum((y_predict - y)**2)

cost minimizer = gradient descent
       partial derivatives of cost funtion for each parameter

'''
import numpy as np
import pandas as pd

data = pd.read_csv('study_hours.csv')

X = (data.Hours).to_numpy()
Y = (data.Scores).to_numpy()
m = data.shape[0]
#modifying X and Y for easy vectorized computation

ones = np.ones(25,)
X = (np.vstack((ones, X))).T
Y = Y.reshape(-1, 1)

#create weight matrix and initialized it with random value

W = np.random.random(X.shape[1]).reshape(-1, 1)
X_data = X[:,1].reshape(-1, 1)
def model(X, W):
    predict = X.dot(W)
    return predict

def mean_squared_error(X, W, Y):
    predict = model(X, W)
    error = 1/2*m * (sum(np.power(predict - Y, 2)))
    return error

def gradient_w0(X, W, Y):
    difference = model(X, W) - Y
    gradient = 1/m * (sum(difference))
    return gradient

def gradient_w1(X, W, Y):
    difference = (model(X, W) - Y) * X_data
    gradient = 1/m * (sum(difference))
    return gradient

def comparer(X, W, Y):
    predict = model(X, W)
    side_by_side = np.hstack((predict, Y))
    return side_by_side
lr = 0.01
initial_error = mean_squared_error(X, W, Y)
def optimizer(type_='batch_gradient'):
    if type_ == 'batch_gradient':
        i = 1
        error = float(mean_squared_error(X, W, Y)[0])
        while error > 1000:
            gradient_0 = gradient_w0(X, W, Y) 
            gradient_1 = gradient_w1(X, W, Y)
            W[0] = W[0] - (lr * gradient_0)
            W[1] = W[1] - (lr * gradient_1)
            error = float(mean_squared_error(X, W, Y)[0])
            print("Error at iteration ", i , "is ", error)
            i += 1
            if i > 100000:
                break

def residual_sum_square():
    predict = model(X, W)
    value = sum(np.power(Y-predict, 2))
    return value

def total_sum_squares():
    mean_ = mean(X)
    normalized = X[:,1] - mean_
    value = sum(np.power(normalized, 2))
    return value
def mean(X):
    return np.mean(X)
def r_squared():
    residual = residual_sum_square()
    total = total_sum_squares()
    value = 1 - ( residual / total )
    return value

optimizer()
#print(initial_error)
#print(mean_squared_error(X, W, Y))

#print(model(X, W))
#print(comparer(X, W, Y ))
print(r_squared()) 



