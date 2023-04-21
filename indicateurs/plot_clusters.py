import os
import time
import numpy as np
import pycxsom as cx
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import sys
import matplotlib.animation as animation
from plot_func import get_tests,get_inputs
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.cluster import KMeans

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

    return df


def plot_inputs(df,weights):
    fig = plt.figure()
    plt.plot(np.linspace(0,1,500),weights['We'])
    plt.scatter(df['bmuM1'],df['I1'])
    plt.scatter(df['bmuM1'],df['I2'])
    plt.scatter(df['bmuM1'],df['I3'])
    plt.show()


def plot_inputs_2D(df,weights,bmu,I):
    fig = plt.figure()
    x =[d[0] for d in df[bmu]]
    y =[d[1] for d in df[bmu]]
    z = df[I]
    print(x)
    print(y)
    print(z)
    plt.scatter(x,y,c = z,cmap='plasma')

if __name__ == "__main__":

    if len(sys.argv)<5:
            sys.exit("Usage: python3 distortion_maps.py <path_to_dir> <test_number> <testclosed> <trie selon> <map names")
    else:
        dir = sys.argv[1]
        test_closed = sys.argv[3]
        test_name = sys.argv[2]
        trie_selon = sys.argv[4]
        map_names = sys.argv[5:]

    dim = len(map_names)
    print(dim)
    test_numbers_0 = int(test_name)
    name = "test-"+str(test_numbers_0)+"-"+str(test_closed)
    df  =read_data("test-"+str(test_numbers_0)+"-"+str(test_closed),test_numbers_0,dir,map_names,trie_selon)
    bmu, wbmu, weight = get_tests(name,test_numbers_0,dir,trie_selon)
    plot_inputs_2D(df,weight,'bmuM2','I1')
    plot_inputs_2D(df,weight,'bmuM2','I2')
    plot_inputs_2D(df,weight,'bmuM2','I3')
    plt.show()
    print(df)
