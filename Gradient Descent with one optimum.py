import matplotlib.pyplot as plt #for plotting
import numpy as np

def function(x):
    return 3*x**2+2*x

x = np.arange(-3.0, 3.0, 0.01)
y = function(x)
plt.plot(x, y,'-r')


x0=float(input('Enter Start Point ')) # learning rate
alpha=float(input('Enter learning rate '))
max_iter=int(input('Enter maximum iteration ')) # No of max iteration
eps0=float(input('Enter precision value '))

X=np.zeros(max_iter+1)
X[0]=x0

Y=np.zeros(max_iter+1)

Y[0] = function(X[0])
Yt = Y[0]

iter = 0 
eps=10

while eps > eps0 and iter < max_iter:
    X[iter+1] = X[iter] - alpha * (6*X[iter]+2)
    Y[iter+1] = function(X[iter+1])
    eps = abs( Yt - Y[iter+1] )
    Yt = Y[iter+1]
    iter = iter + 1

plt.plot(X, Y,'bo',linestyle='-')
print('======================')
print('No of iterations ',iter)
print('X solution',X[iter])
print('Y solution',Y[iter])

plt.show()