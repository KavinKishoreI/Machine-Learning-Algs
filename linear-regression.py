import numpy as np
import matplotlib.pyplot as plt


class LinearRegression :
    def __init__(self, lr = 0.01 , n = 1000):
        self.lr = lr
        self.n = n
        self.weight = 10
        self.bias = 10

    def fit_model (self, x , y):

        X= np.array(x)
        Y= np.array(y)
        if X.shape != Y.shape :
            return "data points missing for some values"

        i = 0
        while ( i < self.n):
            Y_predicted = X * self.weight + self.bias
            der_weights = sum(2 * X * (Y_predicted - Y)) / X.shape[0]
            der_bias = sum(2 * (Y_predicted - Y)) / X.shape[0]

            self.weight = self.weight - self.lr * der_weights
            self.bias = self.bias - self.lr * der_bias

            i+=1

        return float(self.weight) , float(self.bias)
    def predict(self, x):
        if ( self.weight != None and self.bias != None) :
            return  self.weight*x + self.bias

model1 = LinearRegression(0.001 , 5000)
data = [ [3,5,7,10,12] , [5,7,9,10,15]]
slope , intercept = model1.fit_model(*data)
print(model1.predict(6))

x = np.linspace(0, 15, 100)

# Calculate corresponding y-values using the slope-intercept form
y = slope * x + intercept

plt.plot(x, y, label=f'y = {slope}x + {intercept}')
plt.scatter(*data )
plt.show()