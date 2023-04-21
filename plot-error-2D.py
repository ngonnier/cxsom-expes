import os
import time
import numpy as np
import pycxsom as cx
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import math

plt.rc('text', usetex=True)
plt.rcParams['font.size'] = '16'

def bw(weights,pos):
    map_size = weights.shape
    idx = (math.floor(pos[0]*map_size[0]),math.floor(pos[1]*map_size[1]))
    w = weights[idx]
    return w

def get_tests(analysis_prefix,test_number,dir,map_name):
    #get BMUs
    path_bmu = os.path.join(dir,analysis_prefix+'-out',map_name,'BMU.var')
    print(path_bmu)
    with cx.variable.Realize(path_bmu) as fbmu:
        r = fbmu.time_range()
        bmus = np.zeros(r[1],dtype='object')
        for i in range(r[1]):
            bmus[i] = fbmu[i]
    print(map_name)
    w_path = os.path.join(dir,'wgt',map_name)
    #get weights
    files = [f for f in os.listdir(w_path)]
    weights = dict()
    for f in files:
        var_path = os.path.join(w_path,f)
        with cx.variable.Realize(var_path) as w:
            weights_i = w[test_number]
            var_name = f[:-4]
            weights[var_name] = weights_i
    if('We-0' in weights):
        we = weights['We-0']
        map_size=we.shape
        print(map_size)
        #get BMU weights
        if(len(map_size)==1):
            wbmu = [bw(we,elt) for elt in bmus]
        else:
            wbmu=[]
            for elt in bmus:
                bmu_x = int(elt[0]*map_size[0]-1)
                bmu_y = int(elt[1]*map_size[1]-1)
                wbmu.append(we[bmu_x,bmu_y])
    else:
        wbmu = []

    return bmus,wbmu,weights

def get_inputs(dir):
    path_in = os.path.join(dir,"ztest-in")
    files = os.listdir(path_in)
    inputs = dict()
    for inp in files:
        inf= cx.variable.Realize(os.path.join(path_in,inp))
        inf.open()
        r = inf.time_range()
        temp = np.zeros(r[1])
        for i in range(r[1]):
            temp[i] = inf[i]
        inf.close()
        inputs[f'{inp[:-4]}'] = temp
    return inputs


if __name__=="__main__":
        if len(sys.argv)<3:
            sys.exit("usage: python3 plot_error.py <analysis_prefix> <test_number> <data dir> <inputs>")
        analysis_prefix = sys.argv[1]
        tnum = int(sys.argv[2])
        directories = sys.argv[3]
        input_names = sys.argv[4:]
        path = os.path.join(directories, 'wgt')
        maps = os.listdir(path)
        inp_path = os.path.join(directories,'ztest-in')
        bmu1,wbmu1,weights1 = get_tests(analysis_prefix,tnum,directories,"M1")
        bmu2,wbmu2,weights2 = get_tests(analysis_prefix,tnum,directories,"M2")
        inputs = get_inputs(directories)
        if(len(input_names)==2):
            fig,ax = plt.subplots(2,1)
            ax[0].scatter(inputs["I1"],wbmu1)
            ax[1].scatter(inputs["I2"],wbmu2)
            ax[0].set_xlabel('$X^{(1)}$')
            ax[1].set_xlabel('$X^{(2)}$')
            ax[0].set_ylabel('$\\omega_e(\\Pi^{(1)})$')
            ax[1].set_ylabel('$\\omega_e(\\Pi^{(2)})$')

        else:
            if(len(input_names)==3):
                fig,ax = plt.subplots(3,1)
                ax[0].scatter(inputs["I1"],wbmu1)
                ax[1].scatter(inputs["I2"],wbmu2)
                bmu3,wbmu3,weights3 = get_tests(analysis_prefix,tnum,directories,"M3")
                ax[2].scatter(inputs["I3"],wbmu3)
        fig.savefig(f'directories{analysis_prefix}_error.svg')
        plt.show()
