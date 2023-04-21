import pycxsom as cx
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import math
import pandas as pd
from mi_evolution import mutual_information_kraskov
from distortion_maps import read_data,plot,trie

plt.rc('text', usetex=True)
plt.rcParams['font.size'] = '16'

prefix = "4981-"
def get_input_test(dir,input_name):
    path = os.path.join(dir,'input-test',input_name+'.var')
    with cx.variable.Realize(path) as X:
        r = X.time_range()
        x = np.zeros(r[1]+1,dtype=object)
        for i in range(r[1]+1):
            x[i] = X[i]
    return x

def get_input(dir,input_name):
    path = os.path.join(dir,'input',input_name+'.var')
    with cx.variable.Realize(path) as X:
        r = X.time_range()
        x = np.zeros(r[1]+1,dtype=object)
        for i in range(r[1]+1):
            x[i] = X[i]
    return x

def get_test_bmus(dir,test_num,map_name):
    path_bmu = os.path.join(dir,'rlx-test-'+prefix+str(test_num),map_name,'BMU.var')
    print(path_bmu)
    with cx.variable.Realize(path_bmu) as fbmu:
        r = fbmu.time_range()
        bmus = np.zeros(r[1]+1,dtype=object)
        for i in range(r[1]+1):
            bmus[i] = fbmu[i]
    return bmus

def get_weights(dir,map_name):
    w_path = os.path.join(dir,'wgt',map_name)
    files = [f for f in os.listdir(w_path)]
    final_weights = dict()
    for f in files:
        var_path = os.path.join(w_path,f)
        with cx.variable.Realize(var_path) as w:
            r = w.time_range()
            weights = w[r[1]]
            var_name = f[:-4]
            final_weights[var_name] = weights
    return final_weights

def scatter_error(final_weights,bmus,inp,ax,is_closed):
        we =final_weights['We']
        map_size = len(we)
        wbmu = [we[math.floor(elt*map_size)] if(elt<1) else we[-1] for elt in bmus]
        ax.scatter(inp[:len(wbmu)],wbmu)
        if(is_closed):
            ax.set_facecolor("#f7f498")


def plot_time_curve(time,final_weights,bmus,inp,ax,is_closed,bmus_op = None):
    we =final_weights['We']
    map_size = len(we)
    wbmu = [we[math.floor(elt*map_size)] if(elt<1) else we[-1] for elt in bmus]
    ax.plot(time,inp,'k--',label='inputs')
    ax.plot(time,wbmu,'r',label='bmu weights')
    if(is_closed):
        wbmuop = [we[math.floor(elt*map_size)] if(elt<1) else we[-1] for elt in bmus_op]
        ax.plot(time,wbmuop,'b',label='bmu weights closed')



if __name__ == '__main__':

    dir = "corridor_3som"
    map_names = ['MC','MO','MV','MY']
    input_names = ["ct","odomy","vpx","cmdy"]

    inputs = dict()
    for i,m in enumerate(map_names):
        inputs[m] = get_input_test(dir,input_names[i])

    inputs_learn = dict()
    for i,m in enumerate(map_names):
        inputs_learn[m] = get_input(dir,input_names[i])
    #T0
    tests = dict()
    for j in [0,1]:
        print(j)
        bmu = dict()
        for m in map_names:
            bmu[m] = get_test_bmus(dir,j,m)
        tests[j] = bmu

    #T1
    weights= dict()
    for m in map_names:
        weights[m] = get_weights(dir,m)


    #plot weights !
    map_size = len(weights['MY']['We'])
    fig,axes = plt.subplots(2,3)
    for i,m in enumerate(map_names):
        row,col = int(i/3),i%3
        for w in weights[m].items():
            axes[row,col].plot(np.linspace(0,1,map_size),w[1])
            axes[row,col].scatter(tests[0][m],inputs[m])
            axes[row,col].set_title('Carte '+m)


    #remise dans l'ordre :
    permutations = np.fromfile(os.path.join(dir,"permutation_indexes.bin"),  dtype=np.int64)
    print(permutations)
    #pd dataframe avec BMUS, INPUTS
    df = pd.DataFrame(inputs)
    df['idx'] = permutations
    for key,elt in tests.items():
        for name,bmu in elt.items():
            df[f'{name}-{key}'] = bmu

    print(df)
    #class by index !
    df_ord = df.sort_values(by='idx')
    print(df_ord)

    #distortion_maps
    """
    df_weights=read_data('rlx-test-'+prefix+'1',4981,dir,map_names,'MY')
    df_sort_wx = trie(df_weights,'bmu'+'MY')
    f1 = plot(df_sort_wx,'MY',['MY','MC','MV'])
    """
    """
    fig,axes = plt.subplots(2,3)
    scatter_error(weights['MY'],tests[10]['MY'],inputs['MY'],axes[0,0],True)
    #scatter_error(weights['MZ'],tests[10]['MZ'],inputs['MZ'],axes[0,1],True)
    scatter_error(weights['MC'],tests[10]['MC'],inputs['MC'],axes[0,2],False)
    scatter_error(weights['MV'],tests[10]['MV'],inputs['MV'],axes[1,0],False)
    scatter_error(weights['MO'],tests[10]['MO'],inputs['MO'],axes[1,1],False)

    axes[0,0].set_title("error cmdy")
    axes[0,1].set_title("error cmdz")
    axes[0,2].set_title("error ct")
    axes[1,0].set_title("error vpx")
    axes[1,1].set_title("error odomy")

    """
    fig,axes = plt.subplots(2,3)
    """
    plot_time_curve(df_ord['idx'],weights['MY'],df_ord['MY-0'],df_ord['MY'],axes[0,0],True,df_ord['MY-1'])
    plot_time_curve(df_ord['idx'],weights['MZ'],df_ord['MZ-0'],df_ord['MZ'],axes[0,1],True,df_ord['MZ-1'])
    plot_time_curve(df_ord['idx'],weights['MC'],df_ord['MC-1'],df_ord['MC'],axes[0,2],False)
    plot_time_curve(df_ord['idx'],weights['MV'],df_ord['MV-1'],df_ord['MV'],axes[1,0],False)
    plot_time_curve(df_ord['idx'],weights['MO'],df_ord['MO-1'],df_ord['MO'],axes[1,1],False)
    """

    plot_time_curve(df_ord['idx'],weights['MY'],df_ord['MY-0'],df_ord['MY'],axes[0,0],True,df_ord['MY-1'])
    #plot_time_curve(df_ord['idx'],weights['MZ'],df_ord['MZ-0'],df_ord['MZ'],axes[0,1],True,df_ord['MZ-1'])
    plot_time_curve(df_ord['idx'],weights['MC'],df_ord['MC-1'],df_ord['MC'],axes[0,2],False)
    plot_time_curve(df_ord['idx'],weights['MV'],df_ord['MV-1'],df_ord['MV'],axes[1,0],False)
    plot_time_curve(df_ord['idx'],weights['MO'],df_ord['MO-1'],df_ord['MO'],axes[1,1],False)

    axes[0,0].set_title("cmdy")
    axes[0,1].set_title("cmdz")
    axes[0,2].set_title("ct")
    axes[1,0].set_title("vpx")
    axes[1,1].set_title("odomy")

    #plot inputs dependancies
    # fig,axes = plt.subplots(3,3)
    # input_keys= list(inputs_learn.keys())
    # axes[0,0].scatter(inputs_learn[input_keys[0]],inputs_learn[input_keys[1]])
    # axes[0,1].scatter(inputs_learn[input_keys[0]],inputs_learn[input_keys[2]])
    # axes[0,2].scatter(inputs_learn[input_keys[0]],inputs_learn[input_keys[3]])
    # axes[1,0].scatter(inputs_learn[input_keys[1]],inputs_learn[input_keys[2]])
    # axes[1,1].scatter(inputs_learn[input_keys[1]],inputs_learn[input_keys[3]])
    # axes[1,2].scatter(inputs_learn[input_keys[2]],inputs_learn[input_keys[3]])
    #
    # axes[0,0].set_xlabel(input_keys[0])
    # axes[0,0].set_ylabel(input_keys[1])
    # axes[0,1].set_xlabel(input_keys[0])
    # axes[0,1].set_ylabel(input_keys[2])
    # axes[0,2].set_xlabel(input_keys[0])
    # axes[0,2].set_ylabel(input_keys[3])
    # axes[1,0].set_xlabel(input_keys[1])
    # axes[1,0].set_ylabel(input_keys[2])
    # axes[1,1].set_xlabel(input_keys[1])
    # axes[1,1].set_ylabel(input_keys[3])
    # axes[1,2].set_xlabel(input_keys[2])
    # axes[1,2].set_ylabel(input_keys[3])

    fig,axes = plt.subplots(1,3)
    input_keys= list(inputs_learn.keys())
    axes[0].scatter(inputs_learn[input_keys[0]],inputs_learn[input_keys[3]])
    axes[1].scatter(inputs_learn[input_keys[1]],inputs_learn[input_keys[3]])
    axes[2].scatter(inputs_learn[input_keys[2]],inputs_learn[input_keys[3]])
    # axes[1,0].scatter(inputs_learn[input_keys[1]],inputs_learn[input_keys[2]])
    # axes[1,1].scatter(inputs_learn[input_keys[1]],inputs_learn[input_keys[3]])
    # axes[1,2].scatter(inputs_learn[input_keys[2]],inputs_learn[input_keys[3]])
    #
    axes[0].set_xlabel('$\\varphi$')
    axes[0].set_ylabel('$\\rho$')
    axes[1].set_xlabel('$v$')
    axes[1].set_ylabel('$\\rho$')
    axes[2].set_xlabel('$x$')
    axes[2].set_ylabel('$\\rho$')
    # axes[1,0].set_xlabel(input_keys[1])
    # axes[1,0].set_ylabel(input_keys[2])
    # axes[1,1].set_xlabel(input_keys[1])
    # axes[1,1].set_ylabel(input_keys[3])
    # axes[1,2].set_xlabel(input_keys[2])
    # axes[1,2].set_ylabel(input_keys[3])

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    sc = ax.scatter(inputs_learn[input_keys[0]],inputs_learn[input_keys[1]],inputs_learn[input_keys[2]],c=inputs_learn[input_keys[3]])
    ax.set_title(f'{input_keys[0]},{input_keys[1]},{input_keys[2]}')
    plt.colorbar(sc, ax=ax)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(inputs_learn[input_keys[0]],inputs_learn[input_keys[1]],inputs_learn[input_keys[3]])
    ax.set_title(f'{input_keys[0]},{input_keys[1]},{input_keys[3]}')

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(inputs_learn[input_keys[0]],inputs_learn[input_keys[2]],inputs_learn[input_keys[3]])
    ax.set_title(f'{input_keys[0]},{input_keys[2]},{input_keys[3]}')

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(inputs_learn[input_keys[1]],inputs_learn[input_keys[2]],inputs_learn[input_keys[3]])
    ax.set_title(f'{input_keys[1]},{input_keys[2]},{input_keys[3]}')
    """
    plt.show()
