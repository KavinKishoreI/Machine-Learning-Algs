import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
class LogisticRegression:
    e = 2.71828
    def __init__(self, lr=0.001, iter=1000):
        self.lr = lr
        self.iter = iter
        self.weight = None
        self.bias = None
        self.std = 0
        self.mean = 0

    ## gets the x*w + b value
    def sigmoid(self, z):
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )

    def model_fit(self, x, y):
        X = x.to_numpy()
        Y = y.to_numpy()

        start_time = time.time()
        self.mean = X.mean()
        self.std = X.std()

        X =self.feature_scale(X) # scales the large range into a smalll range about [-3 ,  3 ]
        self.weight = 0
        self.bias = 0
        n = X.shape[0]

        for i in range(self.iter):
            z = X * self.weight + self.bias
            prediction = self.sigmoid(z)

            derivative_w = np.dot((prediction - Y), X) / n
            derivative_b = np.sum(prediction - Y) / n
            self.weight -= self.lr * derivative_w
            self.bias -= self.lr * derivative_b
        end = time.time()

        return self.weight, self.bias, end-start_time

    def feature_scale(self, X):
        x = (X - self.mean) / self.std
        return x


    def predict (self, salary ):
        salary = (salary - self.mean) /self.std
        z = salary*self.weight + self.bias
    #threshold is taken as 0.5
        return round(float(self.sigmoid(z)))

model1 = LogisticRegression(lr=0.01, iter=30000)
data = pd.read_csv("Social_Network_Ads.csv")
print(model1.model_fit(data['EstimatedSalary'], data['Purchased']))
print(model1.predict(91000))
plt.scatter(data['EstimatedSalary'] , data['Purchased'])
plt.scatter(data['EstimatedSalary'].to_numpy() , model1.sigmoid(model1.feature_scale(data['EstimatedSalary'])*model1.weight + model1.bias))

plt.show()