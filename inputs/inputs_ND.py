import pycxsom as cx
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import random
import re
cache_size = 2
file_size = 500000

def binom(n, k):
    return math.factorial(n) //( math.factorial(k) * math.factorial(n - k))

def subrotation(angle):
    return np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])

def input_name(name):
    #name = I1, I2 ... to $X^{(1)}$ etc
    if re.match('I\d+',name):
        i_num = re.search('\d+',name)[0]
        return f'$X^{{({i_num})}}$'
    else:
        return f'$X^{{({name})}}$'



class Input:
    #init input from point list
    def __init__(self,name,path,point_list=None,point_test=None):
        self.name = name
        self.path = path
        if point_list is None:
            self.from_file()
        else:
            self.from_points(point_list,point_test)

    def from_points(self,point_list,point_test):
        self.inp = point_list
        self.inp_test = point_test
        self.dim = point_list.shape[1]
        if self.dim ==1:
            self.type = 'Scalar'
        else:
            self.type = f'Array={self.dim}'

    #init input from a file
    def from_file(self):
        with cx.variable.Realize(self.varpath()) as x:
                r = x.time_range()
                X = np.array([x[at] for at in range(r[0],r[1])])
                self.inp = X

        with cx.variable.Realize(self.varpath_test()) as x:
                r = x.time_range()
                X = np.array([x[at] for at in range(r[0],r[1])])
                self.inp_test = X
        if(len(self.inp.shape)>1):
            self.dim = self.inp.shape[1]
        else:
            self.inp= self.inp.reshape(-1,1)
            self.inp_test = self.inp_test.reshape(-1,1)
            self.dim = 1

        if self.dim ==1:
            self.type = 'Scalar'
        else:
            self.type = f'Array={self.dim}'

    def varpath(self):
        return os.path.join(f'{self.path}/in', self.name+'.var')

    def varpath_test(self):
        return os.path.join(f'{self.path}/ztest-in', self.name+'.var')

    def write(self):
        with cx.variable.Realize(self.varpath(), cx.typing.make(self.type),cache_size,file_size) as x:
            for elt in self.inp:
                x += elt
        with cx.variable.Realize(self.varpath_test(), cx.typing.make(self.type),cache_size,file_size) as xtest:
            for elt in self.inp_test:
                xtest += elt


class GeoND:
    def __init__(self,type=None,dim=None,N=None,inputs=None):
        if inputs is None:
            self.points = np.zeros((N,dim))
            self.dim = dim
            self.fill(type)
        else:
            self.dim = sum([i.dim for i in inputs.inp_list])
            nl = inputs.get_nl()
            nt = inputs.get_nt()
            self.N = nl+nt
            self.points = np.zeros((self.N,self.dim))
            sl = 0
            inp_dim = inputs.get_dim()
            for i in inputs.inp_list:
                if i.name == 'U':
                    self.U[0:nl,:] = i.inp
                    self.U[nl:nl+nt,:] = i.inp_test
                else:
                    self.points[0:nl,sl:sl+inp_dim] = i.inp
                    self.points[nl:nl+nt,sl:sl+inp_dim] = i.inp_test
                    sl+=inp_dim

    def fill(self,type):
        if type=='plan':
            self.fill_plan()
            #self.rotation()
        elif type=='sphere':
            self.fill_sphere()
            self.rotation()
        elif type=='hypersphere':
            self.fill_4sphere()
            self.rotation()
        elif type =='lissajoux':
            self.fill_lissajoux()
        elif type=='cy':
            self.fill_cylindre()
            self.rotation()
        elif type == 'cube':
            self.fill_cube()
        elif type == 'cercle':
            self.fill_cercle()
        elif type == 'anneau':
            self.fill_anneau()

    def fill_lissajoux(self):
        rayon = 0.5
        self.U = np.random.rand(len(self.points),1)
        if(self.dim==3):
            A = np.array([[rayon*np.cos(2*2*np.pi *self.U[i]+np.pi/2), rayon*np.sin(3*2*np.pi*self.U[i]), 0] for i in range(self.U.shape[0])])
            alpha = 3*np.pi/4
            beta = np.pi/3
            gamma = np.pi/6
            R_a = np.array([[1,0,0],[0,np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
            R_b = np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0], [-np.sin(beta), 0, np.cos(beta)]])
            R_c = np.array([[np.cos(gamma), -np.sin(gamma), 0],[np.sin(gamma),np.cos(gamma),0], [0,0,1]])
            R_t = np.dot(R_a,R_b).dot(R_c)
            #transformation
            for i  in range(len(self.points)):
                self.points[i,0:3] = [0.5,0.5,0.5] + np.dot(R_t,A[i])

            maxx = np.max(self.points[:,0])
            maxy = np.max(self.points[:,1])
            maxz = np.max(self.points[:,2])

            minx = np.min(self.points[:,0])
            miny = np.min(self.points[:,1])
            minz = np.min(self.points[:,2])


            self.points[:,0] = [(i - minx)/(maxx-minx) for i in self.points[:,0]]
            self.points[:,1] = [(i - miny)/(maxy-miny) for i in self.points[:,1]]
            self.points[:,2] = [(i - minz)/(maxz-minz) for i in self.points[:,2]]
        else:
            self.points = np.array([[0.5 + rayon*np.cos(2*2*np.pi *self.U[i]+np.pi/2), 0.5+rayon*np.sin(3*2*np.pi*self.U[i])] for i in range(self.U.shape[0])])

    def fill_cercle(self):
        rayon = 0.5
        self.U = np.random.rand(len(self.points),1)
        if(self.dim==3):
            A = np.array([[rayon*np.cos(2*np.pi *self.U[i]), rayon*np.sin(2*np.pi*self.U[i]), 0] for i in range(self.U.shape[0])])
            alpha = 3*np.pi/4
            beta = np.pi/3
            gamma = np.pi/6
            R_a = np.array([[1,0,0],[0,np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
            R_b = np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0], [-np.sin(beta), 0, np.cos(beta)]])
            R_c = np.array([[np.cos(gamma), -np.sin(gamma), 0],[np.sin(gamma),np.cos(gamma),0], [0,0,1]])
            R_t = np.dot(R_a,R_b).dot(R_c)
            #transformation
            for i  in range(len(self.points)):
                self.points[i,0:3] = [0.5,0.5,0.5] + np.dot(R_t,A[i])

            maxx = np.max(self.points[:,0])
            maxy = np.max(self.points[:,1])
            maxz = np.max(self.points[:,2])

            minx = np.min(self.points[:,0])
            miny = np.min(self.points[:,1])
            minz = np.min(self.points[:,2])


            self.points[:,0] = [(i - minx)/(maxx-minx) for i in self.points[:,0]]
            self.points[:,1] = [(i - miny)/(maxy-miny) for i in self.points[:,1]]
            self.points[:,2] = [(i - minz)/(maxz-minz) for i in self.points[:,2]]
        else:
            self.points = np.array([[0.5 + rayon*np.cos(2*np.pi *self.U[i]), 0.5+rayon*np.sin(2*np.pi*self.U[i])] for i in range(self.U.shape[0])])

    def fill_anneau(self):
        rayon = 0.5
        epsilon = 0.2
        self.U = np.random.rand(len(self.points),1)
        if(self.dim==3):
            A = np.array([[rayon*np.cos(2*np.pi *self.U[i]), rayon*np.sin(2*np.pi*self.U[i]), 0] for i in range(self.U.shape[0])])
            alpha = 3*np.pi/4
            beta = np.pi/3
            gamma = np.pi/6
            R_a = np.array([[1,0,0],[0,np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
            R_b = np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0], [-np.sin(beta), 0, np.cos(beta)]])
            R_c = np.array([[np.cos(gamma), -np.sin(gamma), 0],[np.sin(gamma),np.cos(gamma),0], [0,0,1]])
            R_t = np.dot(R_a,R_b).dot(R_c)
            #transformation
            for i  in range(len(self.points)):
                self.points[i,0:3] = [0.5,0.5,0.5] + np.dot(R_t,A[i]) + epsilon * np.random.rand(3)

            maxx = np.max(self.points[:,0])
            maxy = np.max(self.points[:,1])
            maxz = np.max(self.points[:,2])

            minx = np.min(self.points[:,0])
            miny = np.min(self.points[:,1])
            minz = np.min(self.points[:,2])


            self.points[:,0] = [(i - minx)/(maxx-minx) for i in self.points[:,0]]
            self.points[:,1] = [(i - miny)/(maxy-miny) for i in self.points[:,1]]
            self.points[:,2] = [(i - minz)/(maxz-minz) for i in self.points[:,2]]
        else:
            self.points = np.array([[0.5 + (rayon-epsilon)*np.cos(2*np.pi *self.U[i]) + epsilon*np.random.rand(), 0.5+(rayon-epsilon)*np.sin(2*np.pi*self.U[i]) + epsilon*np.random.rand()] for i in range(self.U.shape[0])])


    def fill_plan(self):
        self.U = np.random.rand(len(self.points),2)
        if(self.dim==3):
            A = np.array([[self.U[i,0], self.U[i,1],0] for i in range(self.U.shape[0])])
            #self.points = np.reshape(A,(len(self.points),self.dim))
            alpha = 3*np.pi/4
            beta = np.pi/3
            gamma = np.pi/3
            R_a = np.array([[1,0,0],[0,np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
            R_b = np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0], [-np.sin(beta), 0, np.cos(beta)]])
            R_c = np.array([[np.cos(gamma), -np.sin(gamma), 0],[np.sin(gamma),np.cos(gamma),0], [0,0,1]])
            R_t = np.dot(R_a,R_b).dot(R_c)
            # #transformation
            for i  in range(len(self.points)):
                 xi = np.dot(R_t,A[i])
                 self.points[i] = xi
            #
            maxx = np.max(self.points[:,0])
            maxy = np.max(self.points[:,1])
            maxz = np.max(self.points[:,2])

            minx = np.min(self.points[:,0])
            miny = np.min(self.points[:,1])
            minz = np.min(self.points[:,2])

            self.points[:,0] = [(i - minx)/(maxx-minx) for i in self.points[:,0]]
            self.points[:,1] = [(i - miny)/(maxy-miny) for i in self.points[:,1]]
            self.points[:,2] = [(i - minz)/(maxz-minz) for i in self.points[:,2]]
        else:
            self.points = np.copy(self.U)

    def fill_sphere(self):
        N = len(self.points)
        self.U = np.random.rand(N,2)
        rayon = 0.5
        theta = [2*np.pi*self.U[i,1] for i in range(N)]
        phi  = [np.arccos(1 - self.U[i,0]) for i in range(N)]
        A = np.array([[rayon*np.sin(2*phi[i])*np.cos(theta[i]),rayon*np.sin(2*phi[i])*np.sin(theta[i]),rayon*np.cos(2*phi[i])] for i in range(N)])
        self.points[:,0:3] = A
        self.U[:,0] = [2*p/np.pi for p in phi]


    def fill_4sphere(self):
        N = len(self.points)
        self.U = np.random.rand(N,3)
        rayon = 0.5
        #theta = [2*np.pi*self.U[i,1] for i in range(N)]
        self.U[:,0]  = [2*np.arccos(1 - self.U[i,0])/np.pi for i in range(N)]
        self.U[:,1]  = [2*np.arccos(1 - self.U[i,0])/np.pi for i in range(N)]
        A = np.array([[rayon*np.cos(2*np.pi*self.U[i,0]),rayon*np.sin(2*np.pi*self.U[i,0])*np.cos(self.U[i,1]),rayon*np.sin(2*np.pi*self.U[i,0])*np.sin(2*np.pi*self.U[i,1])*np.cos(2*np.pi*self.U[i,2]),rayon*np.sin(2*np.pi*self.U[i,0])*np.sin(2*np.pi*self.U[i,1])*np.sin(2*np.pi*self.U[i,2])] for i in range(N)])
        self.points[:,0:4] = A

    def fill_cube(self):
        N = self.U.shape[0]
        self.U = np.random.rand(N, self.dim)
        self.points = np.copy(self.U)

    def fill_cylindre(self):
        #creation d'un cylindre déja tourné en 3D !!
        rayon = 0.5
        A = np.array([[rayon*np.sin(2*np.pi*self.U[i,0]),rayon*np.cos(2*np.pi*self.U[i,0]),self.U[i,1]] for i in range(self.U.shape[0])])

        alpha = 3*np.pi/4
        beta = np.pi/3
        gamma = np.pi/6
        R_a = np.array([[1,0,0],[0,np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
        R_b = np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0], [-np.sin(beta), 0, np.cos(beta)]])
        R_c = np.array([[np.cos(gamma), -np.sin(gamma), 0],[np.sin(gamma),np.cos(gamma),0], [0,0,1]])
        R_t = np.dot(R_a,R_b).dot(R_c)

        #transformation

        for i  in range(len(self.points)):
            self.points[i,0:3] =  np.dot(R_t,A[i])

    def rotation(self):
        #Translation parameters
        endpoint = 0.5*np.ones(self.dim)
        #random.seed(20)
        random.seed( 6 )
        #-> utilisé pour toutes les seeds !
        #random.seed(14)
        angle = [[2*np.pi*random.random() for i in range(j+1, self.dim)] for j in range(self.dim)]
        #angle = [[0,0,0,np.pi/3,0],[0,0,np.pi/6,0],[np.pi/4,0,0],[0,0],[0],[]]
        #angle = [[1,1,0,0,0] for j in range(self.dim)]
        #angle = [[1,1,0,0,0], [1,1,0,0], [1,1,0,], [1,1], [0], []]
        print(angle)
        dotproduct = np.identity(self.dim)
        for d1 in range(self.dim):
            for d2 in range(d1+1,self.dim):
                print(d1,d2)
                r = np.identity(self.dim)
                s = subrotation(angle[d1][d2-d1-1])
                r[d1,d1] = s[0,0]
                r[d1,(d1+1)%self.dim] = s[0,1]
                r[d2,d2] = s[1,1]
                r[d2,(d2-1)%self.dim] = s[1,0]
                dotproduct = np.dot(dotproduct,r)
        for i  in range(len(self.points)):
            self.points[i,:] = endpoint + np.dot(dotproduct,self.points[i,:])

        #mise à l'échelle [0,1]
        max = np.max(self.points,axis = 0)
        min = np.min(self.points,axis=0)
        for i  in range(self.points.shape[1]):
            if(max[i] - min[i])!=0:
                self.points[:,i] = (self.points[:,i] - min[i])/(max[i] - min[i])

    def plot(self,dim,N,*indices):
        if dim == 3:
            fig=plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            ax.scatter(self.points[0:N,indices[0]], self.points[0:N,indices[1]], self.points[0:N,indices[2]])
            plt.show()
        else:
            fig=plt.figure(size=(10,10))
            ax = fig.add_subplot(3)
            ax[0].scatter(self.points[0:N,indices[0]], self.points[0:N,indices[1]])
            ax[1].scatter(self.points[0:N,indices[0]], self.points[0:N,indices[1]])
            ax[2].scatter(self.points[0:N,indices[0]], self.points[0:N,indices[1]])
            plt.show()


class Inputs:
    def __init__(self,path,nmaps=None,N=None,NS=None,geo=None):
        self.path = path
        self.inp_list = []
        if geo is None:
            #get inputs from path
            varpath = os.path.join(path,'in')
            files = os.listdir(varpath)
            c = 0
            for f in files:
                name = f[0:-4]
                I = Input(name,path)
                if name =='U':
                    self.U = I
                else:
                    c+=1
                    self.inp_list.append(I)
            self.nmaps = c

        else:
            if(N is None or NS is None or nmaps is None):
                raise Exception("Missing Value")
            #get inputs from Geo
            sl = 0
            dim = geo.points.shape[1]
            self.nmaps = nmaps
            input_names = [f'I{i}' for i in range(1,self.nmaps+1)]
            for i in input_names:
                inp = Input(i,path,geo.points[0:N,sl:sl+int(dim/self.nmaps)],geo.points[N:N+NS,sl:sl+int(dim/self.nmaps)])
                self.inp_list.append(inp)
                sl+=int(dim/self.nmaps)
            self.U = Input('U',path,geo.U[0:N],geo.U[N:N+NS])

    def get_nl(self):
        try:
            return len(self.U.inp)
        except AttributeError:
            return len(self.inp_list[0].inp)

    def get_nt(self):
        try:
            return len(self.U.inp_test)
        except AttributeError:
            return len(self.inp_list[0].inp_test)

    def get_dim(self):
        return self.inp_list[0].dim

    #plot inputs from a sample list.
    def plot(self):
        l = len(self.inp_list)
        d = self.get_dim()
        plt.rc('text', usetex=True)
        plt.rcParams['font.size'] = '20'
        if d*l == 2:
            if(d == 1):
                fig,axes = plt.subplots(1,1,figsize=(5,5))
                axes.set_aspect('equal', 'box')
                axes.scatter(self.inp_list[0].inp_test,self.inp_list[1].inp_test)
                i1 = self.inp_list[0].name
                i2 = self.inp_list[1].name
                axes.set_xlabel(input_name(i1))
                axes.set_ylabel(input_name(i2))
            else:
                fig,axes = plt.subplots(1,1)
                axes.scatter(self.inp_list[0].inp_test[:,0],self.inp_list[0].inp_test[:,1])
                i1 = self.inp_list[0].name
                axes.set_xlabel(input_name(i1))
                axes.set_ylabel(input_name(i1))

        elif d*l==3:
            fig  = plt.figure()
            axes = fig.add_subplot(111, projection='3d')
            axes.scatter(self.inp_list[0].inp_test, self.inp_list[1].inp_test, self.inp_list[2].inp_test)
            axes.set_xlabel(input_name(self.inp_list[0].name))
            axes.set_ylabel(input_name(self.inp_list[1].name))
            axes.set_zlabel(input_name(self.inp_list[2].name))

        else:
            if(d == 2):
                fig,axes = plt.subplots(l)
                for i in range(l):
                    axes[i].scatter(self.inp_list[i].inp_test[:,0], self.inp_list[i].inp_test[:,1])
                    axes[i].set_title(input_name(self.inp_list[i].name))
            elif (d==1 and l==4):
                fig,axes = plt.subplots(l//2)
                for i in range(0,l-1,2):
                    axes[i//2].scatter(self.inp_list[i].inp_test, self.inp_list[i+1].inp_test)
                    axes[i//2].set_xlabel(input_name(self.inp_list[i].name))
                    axes[i//2].set_ylabel(input_name(self.inp_list[i+1].name))
            else:
                print(d)
                print(l)
                print("not implemented yet")

        plt.show()


    def write(self):
        for input in self.inp_list:
            input.write()
        self.U.write()

if __name__ == "__main__":
    if(len(sys.argv)<5):
        sys.exit("Usage: python3 inputs.py <path_to_dir> <type> <dim> <n_inputs> <nsamples> <ntest>")
    else:
        path = sys.argv[1]
        try:
            os.system(f"mkdir {path}/ztest-in")
        except FileExistsError:
            pass
        try:
            os.system(f"mkdir {path}/in")
        except FileExistsError:
            pass
    print(sys.argv)
    type = sys.argv[2]
    dim = int(sys.argv[3])
    n_inputs = int(sys.argv[4])
    NSample = int(sys.argv[5])
    Ntest = int(sys.argv[6])

    P = GeoND(type=type,dim=dim, N=NSample+Ntest)
    I =Inputs(path,n_inputs,NSample, Ntest,P)
    I.plot()
    I.write()
