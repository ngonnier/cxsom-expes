import pycxsom as cx
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

cache_size = 2
file_size = 100000

if __name__ == "__main__":
    if(len(sys.argv)<3):
        sys.exit("Usage: python3 inputs.py <path_to_dir> <nb samples> <nb test>")
    else:
        path = sys.argv[1]
        try:
            os.mkdir(f"{path}/in")
            os.mkdir(f"{path}/ztest-in")
        except FileExistsError:
            pass

    def varpath(name):
        return os.path.join(f'{path}/in', name+'.var')

    def varpath_test(name):
        return os.path.join(f'{path}/ztest-in', name+'.var')

    NB_SAMPLES = int(sys.argv[2])
    NB_TEST = int(sys.argv[3])
    t = np.random.rand(NB_SAMPLES+NB_TEST)
    points3D = np.zeros((NB_SAMPLES+NB_TEST,4))
    rayon = 0.5





    for i in range(NB_SAMPLES+NB_TEST):
        #print([rayon*np.cos(2*np.pi*t[i]),rayon*np.sin(2*np.pi*t[i]),0])
        points3D[i,0:3] = [rayon*np.cos(2*np.pi*t[i]),rayon*np.sin(2*np.pi*t[i]),0]
        points3D[i,3] = t[i]

    #No rotation in a sphere

    alpha = np.pi/3
    beta = np.pi/3
    gamma = np.pi/3


    R_a = np.array([[1,0,0],[0,np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    R_b = np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0], [-np.sin(beta), 0, np.cos(beta)]])
    R_c = np.array([[np.cos(gamma), -np.sin(gamma), 0],[np.sin(gamma),np.cos(gamma),0], [0,0,1]])

    R_t = np.dot(R_a,R_b).dot(R_c)

    #Translation parameters
    endpoint = np.array([0.5,0.5,0.5])

    #transformation

    for i  in range(NB_SAMPLES+NB_TEST):
        points3D[i,0:3] = endpoint + np.dot(R_t,points3D[i,0:3])

    #normalization : no  normalization in the sphere ...

    maxx = np.max(points3D[:,0])
    maxy = np.max(points3D[:,1])
    maxz = np.max(points3D[:,2])

    minx = np.min(points3D[:,0])
    miny = np.min(points3D[:,1])
    minz = np.min(points3D[:,2])


    points3D[:,0] = [(i - minx)/(maxx-minx) for i in points3D[:,0]]
    points3D[:,1] = [(i - miny)/(maxy-miny) for i in points3D[:,1]]
    points3D[:,2] = [(i - minz)/(maxz-minz) for i in points3D[:,2]]

    with cx.variable.Realize(varpath('I1'), cx.typing.Scalar(),cache_size, file_size) as x:
        with cx.variable.Realize(varpath('I2'), cx.typing.make('Scalar'),cache_size, file_size) as y:
            with cx.variable.Realize(varpath('I3'), cx.typing.make('Scalar'),cache_size, file_size) as z:
                with cx.variable.Realize(varpath('U'), cx.typing.make('Scalar'),cache_size, file_size) as u:
                    for i in range(NB_SAMPLES):
                        valuex = points3D[i,0]
                        valuey = points3D[i,1]
                        valuez = points3D[i,2]
                        x+=valuex
                        y+=valuey
                        z+=valuez
                        u+=points3D[i,3]


    with cx.variable.Realize(varpath_test('I1'), cx.typing.make('Scalar'),cache_size, file_size) as x:
        with cx.variable.Realize(varpath_test('I2'), cx.typing.make('Scalar'),cache_size, file_size) as y:
            with cx.variable.Realize(varpath_test('I3'), cx.typing.make('Scalar'),cache_size, file_size) as z:
                with cx.variable.Realize(varpath_test('U'), cx.typing.make('Scalar'),cache_size, file_size) as u:
                    for i in range(NB_SAMPLES,NB_SAMPLES+NB_TEST):
                        valuex = points3D[i,0]
                        valuey = points3D[i,1]
                        valuez = points3D[i,2]
                        x+=valuex
                        y+=valuey
                        z+=valuez
                        u+=points3D[i,3]
