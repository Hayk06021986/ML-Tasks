import numpy as np

#Normal equation

X = 2 * np.random.rand(100, 1)
y = X +5+np.random.randn(100, 1)
X_b=np.column_stack((np.ones((100,1)),X))
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)


import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.axis([0, 2.5, 0, 10])

plt.plot(X_b[:,1],X_b.dot(theta),"r",ls='-')
plt.show()

print('theta0=',theta[0],' theta1=',theta[1]) 

def Cost(theta):
    Cost=(1/(2*m))*np.power((X_b.dot(theta) - y),2).sum()
    return Cost


  
#Gradient Method

alpha = 0.01 
n_iterations = 10000
m = 100
theta = np.zeros([2,1]) 
l=[]

for i in range(n_iterations):
    gradient = 1/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - alpha * gradient
    l.append(Cost(theta))
    
plt.plot(l,'r',ls='-')
plt.legend(['Gradient Descent'])
plt.xlabel("Number of iterations")
plt.ylabel("Cost Function")
plt.show()
print('theta0=',theta[0],' theta1=',theta[1]) 

 
    
 #Stochastic Gradient Method   
    
random_index = np.random.randint(m)

theta = np.zeros([2,1])
alpha=0.01
l=[]
for i in range(n_iterations):
    random_index = np.random.randint(100)
    xi = X_b[random_index:random_index+1]
    yi = y[random_index:random_index+1]
    gradients = xi.T.dot(xi.dot(theta) - yi)
    theta = theta - alpha * gradients
    l.append(Cost(theta))

plt.plot(l,'r',ls='-')
plt.legend(['Stochastic Gradient Descent'])
plt.xlabel("Number of iterations")
plt.ylabel("Cost Function")
plt.show()
print('theta0=',theta[0],' theta1=',theta[1]) 
