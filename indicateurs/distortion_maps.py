import os
import time
import numpy as np
import pycxsom as cx
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import sys
import matplotlib.animation as animation
#from plot_func import get_tests,get_inputs
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import scipy.signal as sp

map_size = 500
plt.rc('text', usetex=True)
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

def get_tests(test_name,test_number,dir,map_name):
    #get BMUs
    path_bmu = os.path.join(dir,test_name,map_name,'BMU.var')
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
            wbmu = [we[math.floor(elt*(map_size[0]-1))] for elt in bmus]
        else:
            wbmu=[]
            for elt in bmus:
                bmu_x = int(elt[0]*map_size[0]-1)
                bmu_y = int(elt[1]*map_size[1]-1)
                wbmu.append(we[bmu_x,bmu_y])
    else:
        wbmu = []

    return bmus,wbmu,weights



def read_data(name,idx,dir,map_names,trie_selon):
    n = len(map_names)
    columns = [f'bmu{mname}' for mname in map_names]
    columns += [f'we{mname}' for mname in map_names]
    dfw = pd.DataFrame(columns=columns)
    for i,map in enumerate(map_names):
        bmu, wbmu, weight = get_tests(name,idx,dir,map)
        inp = get_inputs(dir)
        wbmu=np.array(wbmu).flatten()
        bmu=np.array(bmu).flatten()
        dfinp = pd.DataFrame(inp)
        dfw[f'bmu{map}'] = bmu
        dfw[f'we{map}'] = wbmu
        df = dfw.join(dfinp)

    bmu1, wbmu1, weight1 = get_tests(name,idx,dir,trie_selon)
    df[f'bmu{trie_selon}'] = bmu1

    return df

def trie(df,origin):
    extract = df.sort_values(by = origin,axis=0)
    return extract

def plot(df_sort,map,names,dim=3):
    fig=plt.figure()
    if dim==3:
        ax = fig.add_subplot(111,projection='3d')
        ax.plot(df_sort['we'+names[0]],df_sort['we'+names[1]],df_sort['we'+names[2]])
        sc = ax.scatter(df_sort['we'+names[0]],df_sort['we'+names[1]],df_sort['we'+names[2]],c = df_sort[f'bmu{map}'],cmap = 'plasma')
        ax.set_title(f'Map {map}')
        ax.set_xlabel(f'$\\omega_e(\\Pi^{(1)})$')
        ax.set_ylabel(f'$\\omega_e(\\Pi^{(2)})$')
        ax.set_zlabel(f'$\\omega_e(\\Pi^{(3)})$')
        cbar = plt.colorbar(sc)
        cbar.set_label(f'BMU {map}')
    else:
        ax = fig.add_subplot()
        ax.plot(df_sort['we'+names[0]],df_sort['we'+names[1]])
        sc = ax.scatter(df_sort['we'+names[0]],df_sort['we'+names[1]],c = df_sort[f'bmu{map}'],cmap = 'plasma')
        ax.set_title(f'Map {map}')
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        plt.colorbar(sc)
    return fig

def plot_inp(df_sort,map,inpnames,dim=3):
    fig=plt.figure()
    if dim==3:
        ax = fig.add_subplot(111,projection='3d')
        sc = ax.scatter(df_sort[inpnames[0]],df_sort[inpnames[1]],df_sort[inpnames[2]],c = df_sort[f'bmu{map}'],cmap = 'plasma', alpha = 0.2)
        ax.plot(df_sort[inpnames[0]],df_sort[inpnames[1]],df_sort[inpnames[2]])
        ax.set_title(f'Map {map}')
        ax.set_xlabel(inpnames[0])
        ax.set_ylabel(inpnames[1])
        ax.set_zlabel(inpnames[2])
        plt.colorbar(sc)
    else:
        ax = fig.add_subplot()
        ax.plot(df_sort[inpnames[0]],df_sort[inpnames[1]])
        sc = ax.scatter(df_sort[inpnames[0]],df_sort[inpnames[1]],c = df_sort[f'bmu{map}'],cmap = 'plasma')
        ax.set_title(f'Map {map}')
        ax.set_xlabel(inpnames[0])
        ax.set_ylabel(inpnames[1])
        plt.colorbar(sc)
    return fig


def plot_bmus_and_inputs(df_sort,map,names,inpnames,dim=3):
    fig=plt.figure()
    if dim==3:
        ax = fig.add_subplot(111,projection='3d')
        ax.plot(df_sort['we'+names[0]],df_sort['we'+names[1]],df_sort['we'+names[2]])
        sc = ax.scatter(df_sort[inpnames[0]],df_sort[inpnames[1]],df_sort[inpnames[2]],c = df_sort[f'bmu{map}'],cmap = 'plasma')
        sc2 = ax.scatter(df_sort['we'+names[0]],df_sort['we'+names[1]],df_sort['we'+names[2]],c = 'r',s=20,cmap = 'plasma')
        ax.set_title(f'Map {map}')
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.set_zlabel(names[2])
        plt.colorbar(sc)
    else:
        ax = fig.add_subplot()
        ax.plot(df_sort[inpnames[0]],df_sort[inpnames[1]])

        sc = ax.scatter(df_sort[inpnames[0]],df_sort[inpnames[1]],c = df_sort[f'bmu{map}'],cmap = 'plasma',alpha = 0.7)
        sc2 = ax.scatter(df_sort['we'+names[0]],df_sort['we'+names[1]],c = 'r', marker = ',', s=40)
        ax.set_title(f'Map {map}')
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        plt.colorbar(sc)

def plot_bmus(df_sort,map,names,dim=3):
    fig=plt.figure()
    if dim==3:
        ax = fig.add_subplot(111,projection='3d')
        ax.plot(df_sort['bmu'+names[0]],df_sort['bmu'+names[1]],df_sort['bmu'+names[2]])
        sc = ax.scatter(df_sort['bmu'+names[0]],df_sort['bmu'+names[1]],df_sort['bmu'+names[2]],c = df_sort[f'bmu{map}'],cmap = 'plasma')
        ax.set_title(f'Map {map}')
        ax.set_xlabel(f'we{names[0]}')
        ax.set_ylabel(names[1])
        ax.set_zlabel(names[2])
        plt.colorbar(sc)
    else:
        ax = fig.add_subplot()
        ax.plot(df_sort['bmu'+names[0]],df_sort['bmu'+names[1]])
        sc = ax.scatter(df_sort['bmu'+names[0]],df_sort['bmu'+names[1]],c = df_sort[f'bmu{map}'],cmap = 'plasma')
        ax.set_title(f'Map {map}')
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        plt.colorbar(sc)

def plot_jumps(df_sort,names,map):
    if(len(names)==3):
        bmus1 = df_sort['we'+names[0]].to_numpy()
        bmus2 = df_sort['we'+names[1]].to_numpy()
        bmus3 = df_sort['we'+names[2]].to_numpy()
        jumps = []
        last = (0,0,0)

        for w in zip(bmus1,bmus2,bmus3):
            d = distance(last,w)
            jumps.append(d)
            last = w
    else:
        bmus1 = df_sort['we'+names[0]].to_numpy()
        bmus2 = df_sort['we'+names[1]].to_numpy()
        jumps = []
        last = (0,0)

        for w in zip(bmus1,bmus2):
            d = distance(last,w)
            jumps.append(d)
            last = w

    plt.figure()
    plt.plot(jumps)
    plt.title(f"Jumps length in map {map}")
    plt.ylabel("jump amplitude")



def count_units_in_zone(df_sort,map):
    bmus = df_sort[f'bmu{map}'].to_numpy()
    values = set(bmus)
    n_val = len(values)
    gradient = [x-bmus[i-1] if i>0 else 0 for i,x in enumerate(bmus)]
    zones_border = sp.find_peaks(gradient,threshold=0.008)[0]
    np.append(zones_border,int(bmus[-1]*map_size))
    units_in_zone = [(bmus[idx]-bmus[zones_border[i-1]])*map_size if i > 0 else (bmus[idx]-bmus[0])*map_size for i,idx in enumerate(zones_border)]

    plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(bmus[zones_border],units_in_zone)
    plt.title(f'Zones in map {map}')
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.text(0.1,0.1,f'nombres de bmus utilisés : ${n_val}$')
    plt.text(0.1,0.3,f'nombres de zones : {len(zones_border)}')
    plt.text(0.1,0.6,f'taille moyenne d\'une zone: {sum(units_in_zone)/len(units_in_zone)}')
    plt.figure()
    plt.hist(units_in_zone,bins=250)
    plt.title(f'Histogram of units in zones in map {map}')

    return units_in_zone,n_val,len(zones_border),zones_border

def distance(tuple_bmus1,tuple_bmus2):
    n = len(tuple_bmus1)
    d = sum([(elt1-elt2)**2 for elt1,elt2 in zip(tuple_bmus1,tuple_bmus2)])
    d = d/n
    return d


if __name__ == "__main__":
    if len(sys.argv)<5:
            sys.exit("Usage: python3 distortion_maps.py <path_to_dir> <analysis_prefix> <test_number> <testclosed> <trie selon> <map names>")
    else:
        dir = sys.argv[1]
        test_number = sys.argv[3]
        test_closed = sys.argv[4]
        analysis_prefix = sys.argv[2]
        trie_selon = sys.argv[5]
        map_names = sys.argv[6:]

    dim = len(map_names)
    print(dim)
    test_numbers_0 = int(test_number)
    df  =read_data(analysis_prefix+"-out",test_numbers_0,dir,map_names,trie_selon)

    #df_closed_0 = read_data("test-"+str(test_numbers_0)+"-1",test_numbers_0,dir,map_names)
    #df_closed_1 = read_data("test-"+str(test_numbers_0)+"-2",test_numbers_0,dir,map_names)


    if dim == 3:
        df_sortx = trie(df,'bmu'+trie_selon)
        print(df_sortx)

        #1 - Tracer en 3D we0,we1,we2 triées par bmus + tests_closed !!
        f1 = plot(df_sortx,trie_selon,map_names)
        #ax = f1.axes[0]
        #2 Tracer en 3D I1 I2 I3 triés par bmus
        inpnames = ['I'+m[-1] for m in map_names]
        print(inpnames)
        f2 = plot_inp(df_sortx,trie_selon,inpnames)
        #ax = f2.axes[0]
        f3 = plot_bmus_and_inputs(df_sortx,trie_selon,map_names,inpnames)

    else:
        df_sortx = trie(df,'bmu'+trie_selon)
        f1 = plot(df_sortx,trie_selon,map_names,dim=2)
        inpnames = ['I'+m[-1] for m in map_names]
        f2 = plot_inp(df_sortx,trie_selon,inpnames,dim=2)
        #ax = f2.axes[0]
        f3 = plot_bmus_and_inputs(df_sortx,trie_selon,map_names,inpnames,dim=2)



    plt.show()
