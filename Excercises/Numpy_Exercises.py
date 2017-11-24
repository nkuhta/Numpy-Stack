#  libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

###########################################################################
########  Problem #1:  EigenVectors using Euclidean Distance Method  ######
###########################################################################

print('')
print('Problem #1:  EigenVectors using Euclidean Distance Method')
print("**************************************************************")
print('Multiply A*v= vnew, set vnew = v many times')
print('')

#  Starting matrix and vector that adds to 1
A = np.array([[0.3,0.6,0.1],[0.5,0.2,0.3],[0.4,0.1,0.5]])
v = np.array([1/4,1/4,1/2])

# vnew=A.dot(v)
# D = np.sqrt(sum((v-vnew)**2))
# # D=D.sum()
# DE = distance.euclidean(v,vnew)

# number of iterations
NumIter=25

D=np.zeros(NumIter)
x=np.zeros(NumIter)

for i in range(NumIter):
    #  new vector
    vnew = A.dot(v)
    #  euclidean distance to the last v-vector
    DE = distance.euclidean(v,vnew)
    D[i] = DE
    x[i]=i
    #  reset v as the new vnew value
    v = vnew
    print('Distance = ',DE)

print('')
print(' A is: ',A)
print('')

print('')
print('EigenVector for A is: ',v)
print('EigenValue is: ',1)
print('')

plt.plot(x,D)
plt.ylabel('Euclidean Distance')
plt.xlabel('A*v Iteration Number')
plt.show()
#   Note you need to close figure to move onto Problem #2
input('Press Enter to move onto Problem #2...')

###########################################################################
########  Problem #2:  Demonstrate the Central Limit Theorem    ###########
###########################################################################

print('')
print('Problem #2:  Demonstrate the Central Limit Theorem (CLT)')
print("*************************************************************")
print('Find the Gaussian Distribution of the addition of Independent and Identically distributed random values')
print('')

Ylen = 1000

Y=np.zeros(Ylen)

for i in range(Ylen):
    x = np.random.random(1000)
    Y[i]=sum(x)

plt.hist(Y,bins=20)
plt.show()

#   Note you need to close figure to move onto Problem #3
input('Press Enter to move onto Problem #3...')

###########################################################################
########  Problem #3:  Demonstrate the Central Limit Theorem    ###########
###########################################################################
