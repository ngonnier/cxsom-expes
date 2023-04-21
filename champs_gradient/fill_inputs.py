import pycxsom as cx
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

if __name__ == "__main__":
    if(len(sys.argv)<3):
        sys.exit("Usage: python3 inputs.py <path_to_dir> <nb samples> <nb_test>")
    else:
        path = sys.argv[1]
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

        try:
            os.mkdir(f"{path}/in")
            os.mkdir(f"{path}/ztest-in")
        except FileExistsError:
            pass

    def varpath(name):
        return os.path.join(f'{path}/in', name+'.var')

    def varpath_test(name):
        return os.path.join(f'{path}/ztest-in', name+'.var')
        
    print(sys.argv)
    NB_SAMPLES = int(sys.argv[2])
    NB_SAMPLES_TEST = int(sys.argv[3])
    t = np.random.rand(NB_SAMPLES+NB_SAMPLES_TEST)
    points3D = np.zeros((NB_SAMPLES+NB_SAMPLES_TEST,3))
    rayon = 0.5



    for i in range(NB_SAMPLES+NB_SAMPLES_TEST):
        points3D[i,0:2] = [rayon*np.cos(2*np.pi*t[i]),rayon*np.sin(2*np.pi*t[i])]
        points3D[i,2] = t[i]

    #No rotation in a sphere


    #Translation parameters
    endpoint = np.array([0.5,0.5])

    #transformation

    for i  in range(NB_SAMPLES+NB_SAMPLES_TEST):
        points3D[i,0:2] = endpoint + points3D[i,0:2]


    with cx.variable.Realize(varpath('I1'), cx.typing.Scalar(),2,20000) as x:
        with cx.variable.Realize(varpath('I2'), cx.typing.make('Scalar'),2,20000) as y:
                with cx.variable.Realize(varpath('U'), cx.typing.make('Scalar'),2,20000) as u:
                    for i in range(NB_SAMPLES):
                        valuex = points3D[i,0]
                        valuey = points3D[i,1]
                        x+=valuex
                        y+=valuey
                        u+=points3D[i,2]


    with cx.variable.Realize(varpath_test('I1'), cx.typing.make('Scalar'),2,20000) as x:
        with cx.variable.Realize(varpath_test('I2'), cx.typing.make('Scalar'),2,20000) as y:
                with cx.variable.Realize(varpath_test('U'), cx.typing.make('Scalar'),2,20000) as u:
                    for i in range(NB_SAMPLES,NB_SAMPLES+NB_SAMPLES_TEST):
                        valuex = points3D[i,0]
                        valuey = points3D[i,1]
                        x+=valuex
                        y+=valuey
                        u+=points3D[i,2]
