"""

Created on Tue Mar  9 23:11:37 2021

@author: zd187

"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import line_search


def g(x): 
  return np.array([20*x[0], 2*x[1]])

def f(x): 
    return 10*x[0]**2+x[1]**2

def f_plot(x, y):
    return 10*x**2+y**2


x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)

x_star=[0, 0]
x_0=np.array([.2,1])  

plt.figure(figsize=(10,10))

X, Y = np.meshgrid(x, y)
Z = f_plot(X, Y)
cp = plt.contour(X, Y, Z, 50 , cmap=plt.cm.jet, linewidths=0.8)
plt.scatter(x_star[0], x_star[1], color='red', marker=(5, 2))


def SteepestDecent(x_0, epsilon= 0.0001, nIter=2000):
    x=x_0
    nIter=nIter
    epsilon=epsilon
    k=0

    for n in range(nIter):
        if LA.norm(g(x))<epsilon:
            print('g(x)<epsilon')
            break
        
        d=-g(x) 
        alpha=line_search(f, g, x, d)[0]
        x_=x
        x=x+alpha*d

        plt.scatter(x_[0], x_[1], color='b', marker='o',facecolors='none')
        plt.quiver(x_[0], x_[1], alpha*d[0], alpha*d[1], scale_units='xy', angles='xy', scale=1, color='k',linewidths=0.001)
        k+=1
        
    return x

x=SteepestDecent(x_0)
print('Found stationary point of f: ', x)
    
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()