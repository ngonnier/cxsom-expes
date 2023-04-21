import os
import time
import numpy as np
import pycxsom as cx
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import sys
import matplotlib.animation as animation
from plot_func import get_tests
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import scipy.signal as sp

map_size = 100


def read_data(name,idx,dir,map_names):
    n = len(map_names)
    columns = [f'bmu{mname}' for mname in map_names]
    columns += [f'we{mname}' for mname in map_names]
    df = pd.DataFrame(columns=columns)
    for i,map in enumerate(map_names):
        bmu, wbmu, weight = get_tests(name,idx,dir,map)
        wbmu=np.array(wbmu)
        bmu=np.array(bmu)
        bmux = [elt[0] for elt in bmu]
        bmuy = [elt[1] for elt in bmu]

        df[f'bmu{map}'] = bmu
        df[f'we{map}'] = wbmu
        df[f'px{map}'], df[f'py{map}'] = bmux,bmuy

    return df

def trie2D(df,map,mapax):
    extract = df.sort_values(by = mapax+map,axis=0)
    return extract


def plot_mesh(map,names,dim=3):
    #separer la series en plusieurs lignes avec px constant, tracer ces lignes selon py
    #meme chose avec serie triee dans l'autre sens
    df_sort_mapax1 = trie2D(df,map,'px')
    df_sort_mapax2 = trie2D(df,map,'py')
    px = set(df_sort_mapax1[f'px{map}'])
    py = set(df_sort_mapax2[f'py{map}'])
    linesx = []
    linesy = []
    for p in px:
        line = df_sort_mapax1.loc[df_sort_mapax1[f'px{map}'] == p][[f'we{names[0]}',f'we{names[1]}',f'we{names[2]}']].values
        linesx.append(line)

    for p in py:
        line = df_sort_mapax2.loc[df_sort_mapax2[f'py{map}'] == p ][[f'we{names[0]}',f'we{names[1]}',f'we{names[2]}' ]].values
        linesy.append(line)


    fig=plt.figure()
    if dim==3:
        ax = fig.add_subplot(111,projection='3d')
        for l in linesx:
            if len(l)>1:
                plt.plot([elt[0] for elt in l],[elt[1] for elt in l],[elt[2] for elt in l],'b')
        for l in linesy:
            if len(l)>1:
                plt.plot([elt[0] for elt in l],[elt[1] for elt in l],[elt[2] for elt in l],'b')
        #sc = ax.scatter(df_sort['we'+names[0]],df_sort['we'+names[1]],df_sort['we'+names[2]],c = df_sort[f'bmu{map}'],cmap = 'plasma')
        ax.set_title(f'Map {map}')
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.set_zlabel(names[2])

    return fig

def distance(tuple_bmus1,tuple_bmus2):
    n = len(tuple_bmus1)
    d = sum([(elt1-elt2)**2 for elt1,elt2 in zip(tuple_bmus1,tuple_bmus2)])
    d = d/n
    return d


if __name__ == "__main__":
    if len(sys.argv)<4:
            sys.exit("Usage: python3 distortion_maps.py <path_to_dir> <test_number> <n_samp> <map names")
    else:
        dir = sys.argv[1]
        n_samp = int(sys.argv[3])
        test_name = sys.argv[2]
        map_names = sys.argv[4:]

    dim = len(map_names)
    print(dim)
    test_numbers_0 = int(test_name)
    df  =read_data("test-"+str(test_numbers_0)+"-0",test_numbers_0,dir,map_names)

    #df_closed_0 = read_data("test-"+str(test_numbers_0)+"-1",test_numbers_0,dir,map_names)
    #df_closed_1 = read_data("test-"+str(test_numbers_0)+"-2",test_numbers_0,dir,map_names))
    plot_mesh('M1',map_names,dim=3)
    plt.show()
