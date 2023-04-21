from mi_evolution import mutual_information,correlation_ratio,get_inputs,get_bmus, mutual_information_kraskov
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from os.path import isfile, join
import pandas as pd
import pycxsom as cx
import os
from sklearn import svm
from sklearn.linear_model import BayesianRidge
from sklearn.feature_selection import mutual_info_regression
import math

test_times = range(0,9800,200)

def get_tests(test_name,test_number,dir,map_name):
    #get BMUs
    path_bmu = os.path.join(dir,f"zfrz-{test_name}-out",map_name,'BMU.var')
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

def plot_mi_comparison(dir,map_names):
    nfilm = len(test_times)
    mitab_k1 = np.zeros(nfilm)
    mitab_k2 = np.zeros(nfilm)
    minctab_k1 = np.zeros(nfilm)
    minctab_k2 = np.zeros(nfilm)

    xu1_tab = np.zeros(nfilm)
    pix1_tab = np.zeros(nfilm)
    xunc1_tab = np.zeros(nfilm)
    pixnc1_tab = np.zeros(nfilm)
    piw1_tab = np.zeros(nfilm)
    piw2_tab = np.zeros(nfilm)
    xu2_tab = np.zeros(nfilm)
    pix2_tab = np.zeros(nfilm)
    xunc2_tab = np.zeros(nfilm)
    pixnc2_tab = np.zeros(nfilm)

    #path_nc = os.path.join(dir,f'cercle_nc_{xp}')
    #film range
    all_in = get_inputs(dir)
    bins = 250
    x_n = list(map(lambda xx:round(xx*bins)/bins, all_in['I1']))
    y_n = list(map(lambda xx:round(xx*bins)/bins, all_in['I2']))
    imx,eux = mutual_information_kraskov(x_n,all_in['I1'],5)
    imy,euy = mutual_information_kraskov(y_n,all_in['I2'],5)
    #imxn,euxn,_ = mutual_information(x_n,all_in['U'],250,250)
    #imyn,euyn,_ = mutual_information(y_n,all_in['U'],250,250)

    for i,j in enumerate(test_times):
        test_name = '%04d'%j
        print(test_name)
        all_bmus1,wbmus1,w1 = get_tests(test_name,j,dir,"M1")
        all_bmus2,wbmus2,w2 = get_tests(test_name,j,dir,"M2")
        #values
        #imk1,euk1,_= mutual_information(all_bmus['M1'],all_in['U'],250,250)
        #imk2,euk2,_= mutual_information(all_bmus['M2'],all_in['U'],250,250)
        #x1u,eu1,_ = mutual_information(all_in['I1'],all_in['U'],250,250)
        pix1,ex1 = mutual_information_kraskov(all_bmus1,all_in['I1'],5)
        #x2u,eu2,_ = mutual_information(all_in['I2'],all_in['U'],250,250)
        pix2,ex2 = mutual_information_kraskov(all_bmus2,all_in['I2'],5)
        piw1,ew1= mutual_information_kraskov(wbmus1,all_in['I1'],5)
        piw2,ew2 = mutual_information_kraskov(wbmus2,all_in['I2'],5)
        #mitab_k1[i] = imk1
        #mitab_k2[i] = imk2
        #xu1_tab[i] = x1u
        #xu2_tab[i] = x2u
        pix1_tab[i] = pix1
        pix2_tab[i] = pix2
        piw1_tab[i] = piw1
        piw2_tab[i] = piw2
    #plot ces éléments
    fig,ax = plt.subplots(2,1)
    ax[0].set_title('Map X')
    ax[0].set_xlabel('timestep')
    #ax[0].plot(test_times,mitab_k1,'r',label='$I_x(U|\\Pi^{(1)})$')
    ax[0].plot(test_times,pix1_tab,'b',label='$I_x(X^{(1)}|\\Pi^{(1)})$')
    ax[0].plot(test_times,piw1_tab,'r',label='$I_x(X^{(1)}|\\omega(\\Pi^{(1)}))$')
    #ax[0].plot(test_times,xu1_tab,'g',label='$I_x(U|X^{(1)})$')    imyn,euyn,_ = mutual_information(y_n,all_in['U'],250,250)

    ax[0].plot([0,test_times[-1]],[imx,imx],'k--',label='objectif théorique $I_x(X^{(1)}|\\Pi^{(1)})$')
    #ax[0].plot([0,test_times[-1]],[imxn,imxn],'g--',label='objectif théorique $I_x(U|\\Pi)$')
    ax[0].set_ylabel('$I_x$')
    ax[0].legend()

    ax[1].set_title('Map Y')
    ax[1].set_xlabel('timestep')
    #ax[1].plot(test_times,mitab_k2,'r',label='$I_x(U|\\Pi^{(2)})$')
    ax[1].plot(test_times,pix2_tab,'b',label='$I_x(X^{(2)}|\\Pi^{(2)})$')
    ax[1].plot(test_times,piw2_tab,'r',label='$I_x(X^{(2)}|\\omega(\\Pi^{(2)}))$')
    #ax[1].plot(test_times,xu2_tab,'g',label='$I_x(U|X^{(2)})$')
    ax[1].plot([0,test_times[-1]],[imy,imy],'k--',label='objectif théorique $I_x(X^{(2)}|\\Pi^{(2)})$')
    #ax[1].plot([0,test_times[-1]],[imyn,imyn],'g--',label='objectif théorique $I_x(U|\\Pi)$')
    ax[1].set_ylabel('$I_x$')
    ax[1].legend()
    plt.show()

if __name__=='__main__':
    plot_mi_comparison("cercle0",["M1","M2"])
