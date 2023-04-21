import pycxsom as cx
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import re

DEADLINE = 1001

def init_bmu(dir,map,value,test_name):
    with cx.variable.Realize(os.path.join(dir,test_name+'-rlx',map, 'BMU.var'),cx.typing.make('Pos2D'),2,DEADLINE) as bmurlx:
        if(bmurlx.time_range() is None):
            bmurlx[0]= value
        else:
            print('bmu slot already filled')
            pass


def copy_relax_input(dir,input,time_inp,test_name):
    with cx.variable.Realize(os.path.join(dir,test_name+'-in',input+'.var'),cx.typing.Scalar(),1,1) as outrlx:
        with cx.variable.Realize(os.path.join(dir,"ztest-in",input+'.var')) as inprlx:
            if(outrlx.time_range() is None):
                outrlx[0] = inprlx[time_inp]
            else:
                print('input slot already filled')
                pass


if __name__ == "__main__":
    if(len(sys.argv)<5):
        sys.exit("Usage: python3 feed_relax.py <path_to_dir> <prefix> <timestep> <time_inp> <nmaps>")
    else:
        print(sys.argv)
        path = sys.argv[1]
        prefix = sys.argv[2]
        timestep = int(sys.argv[3])
        timeinp = int(sys.argv[4])
        nmaps = int(sys.argv[5])
        try:
            v1 = float(sys.argv[6])
            v2 = float(sys.argv[7])
        except:
            v1 = np.random.rand(2)
            v2 = np.random.rand(2)
            v3 = np.random.rand(2)
            print(v1,v2)

    for i in range(1,nmaps+1):
        #intialiser les bmus pour la relaxation
        init_bmu(path,f'M{i}',v1,prefix)
        copy_relax_input(path,f'I{i}',timeinp,prefix)
