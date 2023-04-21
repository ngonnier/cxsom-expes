import pycxsom as cx
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import re

DEADLINE = 1001

def copy_input(dir,map_target,source,time_inp,test_name):
    with cx.variable.Realize(os.path.join(dir,test_name,map_target,'Input.var'),cx.typing.Scalar(),2,250000) as outrlx:
        with cx.variable.Realize(os.path.join(dir,'ztest-in',source+'.var')) as inprlx:
            if(outrlx.time_range() is None):
                outrlx[0] = inprlx[time_inp]
            else:
                print('input slot already filled')
                pass

def init_bmu(dir,map,value,test_name):
    with cx.variable.Realize(os.path.join(dir,test_name+'-rlx',map, 'BMU.var'),cx.typing.make('Pos1D'),2,DEADLINE) as bmurlx:
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

def str_to_int(n):
    r = r"[1-9]\d*"
    match = re.search(r,n)
    print(match)
    number = match.group(0)
    print(number)
    return int(number)

if __name__ == "__main__":
    if(len(sys.argv)<4):
        sys.exit("Usage: python3 feed_relax.py <path_to_dir> <timestep> <time_inp> <i> <bmu1> <bmu2>")
    else:
        print(sys.argv)
        path = sys.argv[1]
        time_step = sys.argv[2]
        tstep = int(sys.argv[2])
        time_inp = sys.argv[3]
        tinp = int(sys.argv[3])
        i = sys.argv[4]
        try:
            v1 = float(sys.argv[5])
            v2 = float(sys.argv[6])
        except:
            v1 = np.random.rand()
            v2 = np.random.rand()
            print(v1,v2)

    # lancer 1 champ de gradient et 50 expand relax p.e
    #initialiser l'

    test_name = f'out-rlx-{tstep}-{tinp}'
    copy_input(path,'M1','I1',tinp,test_name)
    copy_input(path,'M2','I2',tinp,test_name)

    #intialiser les bmus pour la relaxation
    test_name2 = f'zrlx-{time_step}-{time_inp}-{i}'
    init_bmu(path,'M1',v1,test_name2)
    init_bmu(path,'M2',v2,test_name2)
    copy_relax_input(path,'I1',tinp,test_name2)
    copy_relax_input(path,'I2',tinp,test_name2)
