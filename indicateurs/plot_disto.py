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


map_size = 500


def read_data(name,idx,dir):
    df = pd.DataFrame(columns=['bmux','bmuy','bmuz','wex','wey','wez'])

    bmux, wbmux, weightx = get_tests(name,idx,dir,"M1")
    bmuy, wbmuy, weighty = get_tests(name,idx,dir,"M2")
    bmuz, wbmuz, weightz = get_tests(name,idx,dir,"M3")

    wbmux=np.array(wbmux)
    wbmuy=np.array(wbmuy)
    wbmuz=np.array(wbmuy)
    bmux=np.array(bmux)
    bmuy=np.array(bmuy)
    bmuz=np.array(bmuz)

    df['bmux'] = bmux
    df['bmuy'] = bmuy
    df['bmuz'] = bmuz

    df['wex'] = wbmux
    df['wey'] = wbmuy
    df['wez'] = wbmuz
    print(df)
    return df



#trie la carte 2 en fonction de la carte 1
def trie_3D(bmusx,wbmus):
    list_triee = sorted(zip(bmusx,wbmus),key=lambda tup: tup[0])
    return [tup[1] for tup in list_triee]

def plot_3d(x,y,z,ax,name,color):
    ax.plot(x,y,z,'-o',label=name,c=color)

def count_used(bmus):
    return len(set(bmus))

def zone_size(bmustries):
    tres = 2/500.
    zones_slices = []
    zone = [bmustries[0]]
    for i,elt in enumerate(bmustries[1:]):
        if(elt-bmustries[i-1])>tres:
            zones_slices.append(zone)
            zone = []
        else:
            zone.append(elt)

    return zones_slices,[len(z) for z in zones_slices]


def get_sorted_weights_film(test_names):
    for i,(name,idx) in enumerate(test_names):
        bmux, wbmux, weightx = get_tests(name,idx,dir,"M1")
        bmuy, wbmuy, weighty = get_tests(name,idx,dir,"M2")
        bmuz, wbmuz, weightz = get_tests(name,idx,dir,"M3")
        wbmux=np.array(wbmux).flatten()
        wbmuy=np.array(wbmuy).flatten()
        wbmuz=np.array(wbmuz).flatten()

        tbmux = sorted(bmux[:nb_samp])
        tbmuy = sorted(bmuy[:nb_samp])
        tbmuz = sorted(bmuz[:nb_samp])


        res = dict()
        #trie carte 2 et 3 en fonction de 1
        wx_sx = trie_3D(bmux,wbmux)
        wy_sx = trie_3D(bmux,wbmuy)
        wz_sx = trie_3D(bmux,wbmuz)
        #trie carte 1 et 3 en fonction de 2
        wy_sy = trie_3D(bmuy,wbmuy)
        wx_sy = trie_3D(bmuy,wbmux)
        wz_sy = trie_3D(bmuy,wbmuz)
        #trie carte 1 et 2 en fonction de 3
        wz_sz = trie_3D(bmuz,wbmuz)
        wx_sz = trie_3D(bmuz,wbmux)
        wy_sz = trie_3D(bmuz,wbmuy)

        #Sur une carte 3D, on plot le dépliement de chaque carte avec we(p), wey(wc1(p)), wez(wc2(p))
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(111,projection='3d')
        ax.plot(wx_sx,wy_sx,wz_sx,'r-o',label='Mx')
        ax.plot(wx_sy,wy_sy,wz_sy,'g-o',label='My')
        ax.plot(wx_sz,wy_sz,wz_sz,'b-o',label='Mz')
        ax.legend()
        fig.suptitle(f'Iteration {idx:04d}')
        fig.savefig(os.path.join('./results',path_results,f'film-depliement-{i:03d}.png'))
        plt.close(fig)

def get_sorted_weights_single(test_name,nb_samp):
    name,idx = test_name
    bmux, wbmux, weightx = get_tests(name,idx,dir,"M1")
    bmuy, wbmuy, weighty = get_tests(name,idx,dir,"M2")
    bmuz, wbmuz, weightz = get_tests(name,idx,dir,"M3")
    wbmux=np.array(wbmux).flatten()
    wbmuy=np.array(wbmuy).flatten()
    wbmuz=np.array(wbmuz).flatten()

    usedx = count_used(bmux)
    usedy = count_used(bmuy)
    usedz = count_used(bmuz)

    if nb_samp > len(bmux):
        nb_samp = len(bmux)

    tbmux = sorted(bmux[:nb_samp])
    tbmuy = sorted(bmuy[:nb_samp])
    tbmuz = sorted(bmuz[:nb_samp])

    zonex,zonesizex = zone_size(tbmux)
    zoney,zonesizey = zone_size(tbmuy)
    zonez,zonesizez = zone_size(tbmuz)

    #trie carte 2 et 3 en fonction de 1
    wx_sx = trie_3D(bmux[:nb_samp],wbmux[:nb_samp])
    wy_sx = trie_3D(bmux[:nb_samp],wbmuy[:nb_samp])
    wz_sx = trie_3D(bmux[:nb_samp],wbmuz[:nb_samp])
    #trie carte 1 et 3 en fonction de 2
    wy_sy = trie_3D(bmuy[:nb_samp],wbmuy[:nb_samp])
    wx_sy = trie_3D(bmuy[:nb_samp],wbmux[:nb_samp])
    wz_sy = trie_3D(bmuy[:nb_samp],wbmuz[:nb_samp])
    #trie carte 1 et 2 en fonction de 3
    wz_sz = trie_3D(bmuz[:nb_samp],wbmuz[:nb_samp])
    wx_sz = trie_3D(bmuz[:nb_samp],wbmux[:nb_samp])
    wy_sz = trie_3D(bmuz[:nb_samp],wbmuy[:nb_samp])

    #trie bmus des cartes 2, 3 en fct de 1
    by_sx = trie_3D(bmux[:nb_samp],bmuy[:nb_samp])
    bz_sx = trie_3D(bmux[:nb_samp],bmuz[:nb_samp])

    #trie bmus carte 1 et 3 en fonction de 2
    bx_sy = trie_3D(bmuy[:nb_samp],bmux[:nb_samp])
    bz_sy = trie_3D(bmuy[:nb_samp],bmuz[:nb_samp])
    #trie bmus carte 1 et 2 en fonction de 3
    bx_sz = trie_3D(bmuz[:nb_samp],bmux[:nb_samp])
    by_sz = trie_3D(bmuz[:nb_samp],bmuy[:nb_samp])

    #Sur une carte 3D, on plot le dépliement de chaque carte avec we(p), wey(wc1(p)), wez(wc2(p))
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111,projection='3d')
    ax.plot(wx_sx,wy_sx,wz_sx,'r-',label='Mx')
    ax.plot(wx_sy,wy_sy,wz_sy,'g-',label='My')
    ax.plot(wx_sz,wy_sz,wz_sz,'b-',label='Mz')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.suptitle(f'Iteration {idx:04d}')


    #plot dimension par dimension
    fig,axes= plt.subplots(2,2)
    axes[0,0].plot(wx_sx,wy_sx,'r',label='Mx')
    axes[0,0].plot(wx_sy,wy_sy,'g',label='My')
    axes[0,0].plot(wx_sz,wy_sz,'b',label='Mz')
    axes[0,0].scatter(wx_sz,wy_sz,color='k')
    axes[0,0].legend()
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    axes[1,0].plot(wx_sx,wz_sx,'r',label='Mx')
    axes[1,0].plot(wx_sy,wz_sy,'g',label='My')
    axes[1,0].plot(wx_sz,wz_sz,'b',label='Mz')
    axes[1,0].scatter(wx_sz,wz_sz,color='k')
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('z')
    axes[0,1].plot(wy_sx,wz_sx,'r',label='Mx')
    axes[0,1].plot(wy_sy,wz_sy,'g',label='My')
    axes[0,1].plot(wy_sz,wz_sz,'b',label='Mz')
    axes[0,1].scatter(wy_sz,wz_sz,color='k')
    axes[0,1].set_xlabel('y')
    axes[0,1].set_ylabel('z')
    fig.suptitle(f'Iteration {idx:04d}')
    #fig.savefig(os.path.join('./results',path_results,f'film-depliement-{i:03d}.png'))

    fig = plt.figure()
    ax0 = fig.add_subplot(111,projection='3d')
    ax0.plot(wx_sx,wy_sx,wz_sx,'r-')
    sc=ax0.scatter(wx_sx,wy_sx,wz_sx,c=tbmux,cmap='plasma')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.set_zlabel('z')
    ax0.set_title('Mx')
    plt.colorbar(sc)
    fig = plt.figure()
    ax0 = fig.add_subplot(111,projection='3d')
    ax0.plot(wx_sy,wy_sy,wz_sy,'g-')
    sc=ax0.scatter(wx_sy,wy_sy,wz_sy,c=tbmuy,cmap='plasma')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.set_zlabel('z')
    ax0.set_title('My')
    plt.colorbar(sc)
    fig = plt.figure()
    ax0 = fig.add_subplot(111,projection='3d')
    ax0.plot(wx_sz,wy_sz,wz_sz,'b-')
    sc=ax0.scatter(wx_sz,wy_sz,wz_sz,c=tbmuz,cmap='plasma')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.set_zlabel('z')
    ax0.set_title('Mz')
    plt.colorbar(sc)
    #plot BMUs in 3d

    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111,projection='3d')
    ax.plot(tbmux,by_sx,bz_sx,'r-o',label='Mx')
    ax.plot(bx_sy,tbmuy,wz_sy,'g-o',label='My')
    ax.plot(bx_sz,by_sz,tbmuz,'b-o',label='Mz')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.suptitle(f'Iteration {idx:04d}')

    fig = plt.figure()
    ax0 = fig.add_subplot(111,projection='3d')
    ax0.plot(tbmux,by_sx,bz_sx,'r-')
    sc=ax0.scatter(tbmux,by_sx,bz_sx,c=tbmux,cmap='plasma')
    #ax0.scatter(bmux,wy_sx,wz_sx,'r-o')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.set_zlabel('z')
    ax0.set_title('Mx')
    plt.colorbar(sc)
    fig = plt.figure()
    ax0 = fig.add_subplot(111,projection='3d')
    ax0.plot(bx_sy,tbmuy,bz_sy,'g-')
    sc=ax0.scatter(bx_sy,tbmuy,bz_sy,c=tbmuy,cmap='plasma')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.set_zlabel('z')
    ax0.set_title('My')
    plt.colorbar(sc)
    fig = plt.figure()
    ax0 = fig.add_subplot(111,projection='3d')
    ax0.plot(bx_sz,by_sz,tbmuz,'b-')
    sc=ax0.scatter(bx_sz,by_sz,tbmuz,c=tbmuz,cmap='plasma')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.set_zlabel('z')
    ax0.set_title('Mz')
    plt.colorbar(sc)
    """
    #plot zone elemnts on map disto to check if correct
    print(zonex)
    firstlast = sorted([z[0] for z in zonex if len(z)>0]+[z[-1] for z in zonex if len(z)>0])
    firstlasty = [y for x,y in zip(tbmux,by_sx) if x ]
    firstlastz = [z for x,z in zip(tbmux,bz_sx) if x in firstlast]

    print(len(firstlasty))
    print(len(firstlast))

    fig = plt.figure()
    ax0 = fig.add_subplot(111,projection='3d')
    ax0.scatter(tbmux,by_sx,bz_sx,color='k')
    ax0.scatter(firstlast,firstlasty,firstlastz,color='r')
    ax0.plot(tbmux,by_sx,bz_sx,'r-')
    """
if __name__=="__main__":
    test_name = ""
    if len(sys.argv)<5:
            sys.exit("Usage: python3 plot_disto.py <path_to_dir> <path_to_result> <test_name> <n_samp>")
    else:
        dir = sys.argv[1]
        path_results = sys.argv[2]
        n_samp = int(sys.argv[4])
        test_name = sys.argv[3]

    map_names = ["M1","M2","M3"]
    input_names = ["I1","I2","I3"]
    inputs = np.zeros((100,3))
    test_numbers_0 = int(test_name)
    df  =read_data("test-"+str(test_numbers_0)+"-0",test_numbers_0,dir)


    """
    path_inp = os.path.join(dir,"input-test")
    for i in range(len(input_names)):
        with cx.variable.Bind(os.path.join(path_inp,input_names[i]+'.var')) as inp:
            for j in range(100):
                inputs[j,i] = inp[j]
    """
    #tests_numbers_0 = list(range(0,500,10))
    #tests_numbers_1 = list(range(500,3000,100))
    #tests_numbers_2 =list(range(3000,19000,500))
    #test_numbers = sorted(tests_numbers_0+tests_numbers_1+tests_numbers_2)
    test_numbers = range(0,20000,200)

    if test_name == "film":
        test_names_film = [("test-film-"+str(tidx),tidx) for tidx in test_numbers]
        get_sorted_weights_film(test_names_film)
    else:
        test_name_0 = ("test-"+str(test_numbers_0)+"-0",test_numbers_0)
        get_sorted_weights_single(test_name_0,n_samp)
        plt.show()
