import numpy as np
import matplotlib.pyplot as plt
from colormap import *
MAP_SIZE = 100
import random
plt.rc('text', usetex=True)
plt.rcParams['font.size'] = '16'
DIM = 4



def binom(n, k):
    return math.factorial(n) //( math.factorial(k) * math.factorial(n - k))

def subrotation(angle):
    return np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])


# création d'une sphere 3D
N = 2500
points4D = np.zeros((N,4))
U = np.random.rand(N,2)
rayon = 0.5
theta = [2*np.pi*U[i,1] for i in range(N)]
phi  = [np.arccos(1 - U[i,0]) for i in range(N)]

colormap2D = np.array([[[i/MAP_SIZE, j/MAP_SIZE] for j in range(MAP_SIZE)] for i in range(MAP_SIZE)])
cmap = colormap_a(colormap2D,'ziegler')
print(cmap.shape)
A = np.array([[rayon*np.sin(2*phi[i])*np.cos(theta[i]),rayon*np.sin(2*phi[i])*np.sin(theta[i]),rayon*np.cos(2*phi[i])] for i in range(N)])
points4D[:,0:3] = A
U[:,0] = [2*p/np.pi for p in phi]

#plot 3D sphere with U as correlation_ratio
fig3D = plt.figure()
ax = fig3D.add_subplot(131, projection='3d')
ax.scatter(points4D[:,0],points4D[:,1],points4D[:,2], c=colormap_l(U,type='ziegler',n=len(points4D[:,0])))
uv = fig3D.add_subplot(132)
uv.scatter(U[:,0],U[:,1],c=colormap_l(U,type='ziegler',n=N))
uv.set_xlabel('$\\phi$')
uv.set_ylabel('$\\theta$')
cmapf = fig3D.add_subplot(133)
cmapf.imshow(cmap,extent=(0,1,0,1),origin='lower')
cmapf.set_xlabel('$\\phi$')
cmapf.set_ylabel('$\\theta$')

figsp = plt.figure()
ax = figsp.add_subplot(projection='3d')
ax.scatter(points4D[:,0],points4D[:,1],points4D[:,2], c=colormap_l(U,type='ziegler',n=len(points4D[:,0])))
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
#rotation en 4D et 6D
#Translation parameters
endpoint = 0.5*np.ones(DIM)
#random.seed(20)
random.seed(6)
angle = [[2*np.pi*random.random() for i in range(j+1, DIM)] for j in range(DIM)]
#angle = [[np.pi/3 for i in range(j+1, DIM)] for j in range(DIM)]
print(angle)
dotproduct = np.identity(DIM)
for d1 in range(DIM):
    for d2 in range(d1+1,DIM):
        r = np.identity(DIM)
        s = subrotation(angle[d1][d2-d1-1])
        r[d1,d1] = s[0,0]
        r[d1,(d1+1)%DIM] = s[0,1]
        r[d2,d2] = s[1,1]
        r[d2,(d2-1)%DIM] = s[1,0]
        dotproduct = np.dot(dotproduct,r)
        print(r)
for i  in range(len(points4D)):
    points4D[i,:] = endpoint + np.dot(dotproduct,points4D[i,:])

#mise à l'échelle [0,1]
max = np.max(points4D,axis = 0)
min = np.min(points4D,axis=0)
for i  in range(points4D.shape[1]):
    if(max[i] - min[i])!=0:
        points4D[:,i] = (points4D[:,i] - min[i])/(max[i] - min[i])

#plot 4D sphere with U as colormap
fig4D = plt.figure()
ax = fig4D.add_subplot(131)
ax.scatter(points4D[:,0],points4D[:,1], c=colormap_l(U,type='ziegler',n=len(points4D[:,0])))
ax.set_title('$X^{(1)}$')
ax.set_xlabel('$X^{(1)}|_x$')
ax.set_ylabel('$X^{(1)}|_y$')
ax = fig4D.add_subplot(132)
ax.scatter(points4D[:,2],points4D[:,3], c=colormap_l(U,type='ziegler',n=len(points4D[:,0])))
ax.set_title('$X^{(2)}$')
ax.set_xlabel('$X^{(2)}|_x$')
ax.set_ylabel('$X^{(2)}|_y$')
# uv = fig.add_subplot(133)
# uv.scatter(U[:,0],U[:,1],c=colormap_l(U,type='ziegler',n=N))
# uv.set_xlabel('$\\phi$')
# uv.set_ylabel('$\\theta$')
# cmapf = fig4D.add_subplot(144)
# cmapf.imshow(cmap,extent=(0,1,0,1),origin='lower')
# cmapf.set_xlabel('$\\phi$')
# cmapf.set_ylabel('$\\theta$')
# cmapf.set_title('$U$')
uv = fig4D.add_subplot(133)
uv.scatter(U[:,0],U[:,1],c=colormap_l(U,type='ziegler',n=N))
uv.set_xlabel('$\\phi$')
uv.set_ylabel('$\\theta$')
uv.set_xlabel('$\\phi$')
uv.set_ylabel('$\\theta$')
uv.set_title('$U$')
plt.show()
