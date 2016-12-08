import numpy as np
import time

start_time = time.time()

X = np.loadtxt(open("logistic_x.txt"))

temp = np.ones((1,X.shape[0]))
X = np.hstack((X,np.atleast_2d(temp).T))

Y = np.loadtxt(open("logistic_y.txt"))

m = X.shape[0]

Theta = np.zeros(X[0].shape)

def h(theta, x):
	z = np.dot(x, theta)
	g = 1 / ( 1 + np.exp(-z))
	return g

d = np.ones(Theta.shape)

while True:
    H = 0
    d = np.zeros(X[0].shape)
    for i in range(m):
        n  = np.linalg.norm(X[i]) ** 2
        H += (n)*(h(Theta, X[i]))*(1 - h(Theta, X[i]))
        d += (h(Theta, X[i])-Y[i])*X[i]
    H /= m
    d /= m
    d /= H

    Theta -= d

    if time.time() - start_time > 10: break

print(Theta)