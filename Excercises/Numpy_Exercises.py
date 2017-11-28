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
    vnew = np.dot(A,v)
    #  euclidean distance to the last v-vector
    DE = distance.euclidean(v,vnew)
    #DE = np.sqrt(sum((v-vnew)**2))
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
print('Finding the Gaussian Distribution from the addition of ')
print('Independent and Identically distributed random values')
print('')

Ylen = 1000

Y=np.zeros(Ylen)

for i in range(Ylen):
    x = np.random.random(1000)
    Y[i]=sum(x)

plt.hist(Y,bins=20)
plt.show()

#   Note you need to close figure to move onto Problem #5
input('Press Enter to move onto Problem #5...')

###########################################################################
########     Problem #5:  Is a Matrix Symmetric Function        ###########
###########################################################################

print('')
print('Problem #5:  Is a Matrix Symmetric?  ie. M == M_Transpose ')
print("*************************************************************")
print('write and use is_symmetric() function')
print('')

#  Returns True if matrix is Symmetric and False if not-symmetric
def is_symmetric(M):
    return np.array_equal(M,M.transpose())

A1 = np.array([[1,2],[2,1]])
A2 = np.array([[1,2],[3,4]])

print('A1 = ',A1)
print('')
print('A2 = ',A2)
print('')

#print(np.array_equal(A2,A2.transpose()))

print('Is A1 Symmetric: ',is_symmetric(A1))
print('Is A2 Symmetric: ',is_symmetric(A2))
print('')

#   Note you need to close figure to move onto Problem #6
input('Press Enter to move onto Problem #6...')

###########################################################################
########    Problem #6:  Generate and Plot the XOR Dataset      ###########
###########################################################################

print('')
print('Problem #6:  Generate and Plot XOR Dataset ')
print("*************************************************************")
print('')

#  first variable (x1) data points between (-1,1)
x1 = 2*np.random.random(1500)-1
x2 = 2*np.random.random(1500)-1
y=np.logical_xor(np.ceil(x1),np.ceil(x2))

plt.scatter(x1,x2,c=y,cmap='cool',edgecolor='black')
plt.show()

###########################################################################
##########    Problem #7:  2 Concentric Circles Dataset       #############
###########################################################################

print('')
print('Problem #7:  Generate/Plot 2 Circles around R=10,20 with R_noise = +/- 1 ')
print("**************************************************************************")
print('')

#  Creat polar r=10 and R=20 data for all theta values
r1 = 2*np.random.random(200)+9
r2 = 2*np.random.random(200)+19
theta1 = 2*np.pi*np.random.random(200)

#ax = plt.subplot(projection='polar')
# plt.scatter(r1,theta1)
# plt.show()
#fig = plt.figure()
#ax = fig.add_subplot(111, polar=True)
#c = ax.scatter(theta1, r1)
#fig.show()

#  Equation for X using R/theta, remember the squart root negative sign :)
x = np.sqrt(r1**2/(1+np.tan(theta1)**2))
#  Here to account for negative x, just mirror in the case.
xneg = -1*x
#  same equation for y, except multiplied by tan(theta)
y = np.tan(theta1)*np.sqrt(r1**2/(1+np.tan(theta1)**2))
yneg = -1*y
x2 = np.sqrt(r2**2/(1+np.tan(theta1)**2))
#  Here is the second, larger radius
x2neg = -1*x2
y2 = np.tan(theta1)*np.sqrt(r2**2/(1+np.tan(theta1)**2))
y2neg = -1*y2
#  R1 x values
xtot=np.append(x,xneg)
#  R1 y values
ytot=np.append(y,yneg)
#  R2 x values
x2tot=np.append(x2,x2neg)
#  R2 y values
y2tot=np.append(y2,y2neg)
#  scatter plot for both R1, and R2
plt.scatter(xtot,ytot)
plt.scatter(x2tot,y2tot)
#  Set axis limits, and equal aspect ratio
plt.xlim(-30,30)
plt.ylim(-30,30)
plt.axes().set_aspect('equal')
plt.show()


###########################################################################
##########      Problem #8:  Mutiple Sprirals Dataset         #############
###########################################################################

print('')
print('Problem #8:  Generate/Plot Mutiple Spirals ')
print("********************************************")
print('')

#  Cartesian X-value function
def xcart(r,theta):
    return np.sqrt(r**2/(1+np.tan(theta)**2))

#  Cartesian Y-value function
def ycart(r,theta):
    return np.tan(theta)*np.sqrt(r**2/(1+np.tan(theta)**2))

tsteps=200

#  Here is a brute force method without thoughtful colar scheme
#  Six sets of theta values all stepping in time.
#  Reuse the same R series for all theta

r1 = np.linspace(0.1,10,200)#+np.random.random(200)
theta1=np.linspace(0,2*np.pi/6,tsteps)
theta2=np.linspace(2*np.pi/6,4*np.pi/6,tsteps)
theta3=np.linspace(4*np.pi/6,6*np.pi/6,tsteps)
theta4=np.linspace(6*np.pi/6,8*np.pi/6,tsteps)
theta5=np.linspace(8*np.pi/6,10*np.pi/6,tsteps)
theta6=np.linspace(10*np.pi/6,12*np.pi/6,tsteps)

#  6 (x,y) series from the polar data, with Cartesian noise added on
x1=xcart(r1,theta1)+np.random.random(tsteps)
y1=ycart(r1,theta1)+np.random.random(tsteps)
x2=xcart(r1,theta2)+np.random.random(tsteps)
y2=ycart(r1,theta2)+np.random.random(tsteps)
x3=xcart(r1,theta3)+np.random.random(tsteps)
y3=ycart(r1,theta3)+np.random.random(tsteps)
x4=-xcart(r1,theta4)+np.random.random(tsteps)
y4=-ycart(r1,theta4)+np.random.random(tsteps)
x5=-xcart(r1,theta5)+np.random.random(tsteps)
y5=-ycart(r1,theta5)+np.random.random(tsteps)
x6=-xcart(r1,theta6)+np.random.random(tsteps)
y6=-ycart(r1,theta6)+np.random.random(tsteps)

#  plot it all, note nice color scheme isn't figured out here yet.
plt.scatter(x1,y1,cmap='seismic')
plt.scatter(x2,y2,cmap='seismic')
plt.scatter(x3,y3,cmap='seismic')
plt.scatter(x4,y4,cmap='seismic')
plt.scatter(x5,y5,cmap='seismic')
plt.scatter(x6,y6,cmap='seismic')
plt.show()

###########################################################################
##########      Problem #9:  Export XOR Dataframe to CSV      #############
###########################################################################

print('')
print('Problem #9:  Export XOR Dataframe to CSV')
print("********************************************")
print('')

# xor data 
x1 = 2*np.random.random(150)-1
x2 = 2*np.random.random(150)-1
y=np.logical_xor(np.ceil(x1),np.ceil(x2))*1

#  roll up all data into a dictionary with column lables and value lists
dic={'x1':x1,"x2":x2,"y":y}
#  create the dataframe from the dictionary
df = pd.DataFrame(dic)

print(df)

#  write dataframe to csv without the first index column
df.to_csv('xor.csv',index=False)
