import os
import time
import numpy as np
import pycxsom as cx
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import sys
import struct
from plot_func import *
from scipy.stats import entropy

def correlation_ratio(U,P,n_u,n_p):
    phiY = np.zeros((n_p,1))
    hist_up,pedge,uedge = np.histogram2d(P,U,bins = [n_p,n_u])
    hist_u = np.sum(hist_up, axis = 0)
    hist_p = np.sum(hist_up, axis = 1)

    for i in range(n_p):
        if hist_p[i] == 0:
            phiY[i] = 0
        else:
            phiY[i] = 1/hist_p[i] * sum([uedge[j]*hist_up[i,j] for j in range(n_u)])

    cr = np.var(phiY)/np.var(U)

    return cr,phiY


def correlation_ratio_ND(U,P,n_p):
    histo = np.zeros((n_p,n_p))
    moyenne = np.zeros((n_p,n_p,2))
    phiY = np.zeros((n_p,n_p,2))
    for i in range(5999):
        pos = P[i]
        u = U[i]
        ipos0 = int((n_p-1)*(pos[0]))
        ipos1 = int((n_p-1)*(pos[1]))
        histo[ipos0,ipos1] = histo[ipos0,ipos1] + 1
        moyenne[ipos0,ipos1,:] += u


    for i in range(n_p):
        for j in range(n_p):
            if histo[i,j] == 0:
                phiY[i,j] = 0
            else:
                phiY[i,j,:] = 1/histo[i,j] * moyenne[i,j]

    print(U.shape)
    print(phiY.shape)
    U = np.array(U)
    return cr,phiY


def mutual_information(bmu_pos,U,n_u,n_p):
    nb_val = len(U)
    hist_up,_,_ = np.histogram2d(bmu_pos,U,bins = [n_p,n_u])
    hist_u,_ = np.histogram(U,bins=n_u)
    hist_p,_ = np.histogram(bmu_pos,bins=n_p)
    eu = entropy(hist_u)
    ep = entropy(hist_p)
    hist_up = hist_up.flatten()
    epu = entropy(hist_up)
    im = eu + ep - epu
    return im,eu

def mi_conj_n(p,u,n_u,n_p):
    up = np.empty((p.shape[0],p.shape[1]+u.shape[1]))
    up[:,0:p.shape[1]] = p
    up[:,-u.shape[1]] = u
    print(up.shape)
    hist_upp,edges = np.histogramdd(up,bins = [n_p]*p.shape[1])+[n_u]*u.shape[1])
    hist_u,_ = np.histogramdd(u,bins=n_u)
    hist_pp,_ = np.histogramdd(p,bins=n_p)
    hist_u = hist_u.flatten()
    hist_pp = hist_pp.flatten()
    hist_upp = hist_upp.flatten()
    eu = entropy(hist_u)
    epu = entropy(hist_upp)
    ep = entropy(hist_pp)
    im = eu + ep - epu
    return im,eu

if __name__=="__main__":
    if len(sys.argv)<3:
        sys.exit("Usage: python3 correlation_ratio.py <path_to_dir> <prefix>")
    else:
        dir = sys.argv[1]
        prefix = sys.argv[2]
    #importer inputs tests :
    in_path = os.path.join(dir,'input-test')
    files = [f for f in os.listdir(in_path)]
    all_in = dict()
    for f in files:
        var_path = os.path.join(in_path,f)
        input = []
        with cx.variable.Realize(var_path) as inp:
            print(inp.time_range())
            for i in range(inp.time_range()[1]):
                input.append(inp[i])
            var_name = f[:-4]
            all_in[var_name]=input

    keys  = list(all_in.keys())
    #calcul correlation_ratio #balec
    """
    corr_ratio_12,phi12 = correlation_ratio(all_in['I1'],all_in['I2'],100,100)
    corr_ratio_13,phi13 = correlation_ratio(all_in['I1'],all_in['I3'],100,100)
    corr_ratio_23,phi23 = correlation_ratio(all_in['I2'],all_in['I3'],100,100)
    corr_ratio_1U,phi1U = correlation_ratio(all_in['I1'],all_in['U'],100,100)
    corr_ratio_2U,phi2U = correlation_ratio(all_in['I2'],all_in['U'],100,100)
    corr_ratio_3U,phi3U = correlation_ratio(all_in['I3'],all_in['U'],100,100)
    """
    #importer BMUS de tests
    map_name = 'M1'
    input_name = 'X'
    bmu,wbmu,weights = get_tests(f'test-{prefix}-0',18999,dir,map_name)

    mi_1u,e1_u = mutual_information(bmu,all_in['U'],100,100)
    #mi_2u,e2_u = mutual_information(map_data[1].test_bmus,map_data[1].test_u,100,100)
    #mi_3u,e3_u = mutual_information(map_data[2].test_bmus,map_data[2].test_u,100,100)
    m1 = mi_1u/e1_u
    print(m1)
    corr_ratio_B1U, phiB1U = correlation_ratio(all_in['U'],bmu,500,500)
    #corr_ratio_B2U, phiB2U = correlation_ratio(map_data[1].test_u,map_data[1].test_bmus,100,100)
    #corr_ratio_B3U, phiB3U = correlation_ratio(map_data[2].test_u,map_data[2].test_bmus,100,100)

    #plot BMU weights and tests
    #print(weights['We'])
    plt.figure()
    plt.plot([w[0] for w in weights['We']],[w[1] for w in weights['We']] )


    plt.figure()
    plt.scatter(bmu,all_in['U'])
    plt.plot(np.linspace(0,1,len(phiB1U)),phiB1U,'r')


    """
    plt.figure()
    plt.subplot(111,projection='3d')
    plt.scatter(all_in['I1'],all_in['I2'], all_in['I3'])
    plt.figure()
    plt.subplot(2,3,1)
    plt.scatter(all_in['I2'],all_in['I1'])
    plt.plot(np.linspace(0,1,len(phi12)),phi12,'r')
    plt.title(f"correlation ratio:{corr_ratio_12}")
    plt.xlabel('I1')
    plt.ylabel('I2')

    plt.subplot(2,3,2)
    plt.scatter(all_in['I3'],all_in['I2'])
    plt.plot(np.linspace(0,1,len(phi23)),phi23,'r')
    plt.title(f"correlation ratio:{corr_ratio_23}")
    plt.xlabel('I2')
    plt.ylabel('I3')

    plt.subplot(2,3,3)
    plt.scatter(all_in['I3'],all_in['I1'])
    plt.plot(np.linspace(0,1,len(phi13)),phi13,'r')
    plt.title(f"correlation ratio:{corr_ratio_13}")
    plt.xlabel('I1')
    plt.ylabel('I3')

    plt.subplot(2,3,4)
    plt.scatter(all_in['U'],all_in['I1'])
    plt.plot(np.linspace(0,1,len(phi1U)),phi1U,'r')
    plt.title(f"correlation ratio:{corr_ratio_1U}")
    plt.xlabel('I1')
    plt.ylabel('U')

    plt.subplot(2,3,5)
    plt.scatter(all_in['U'],all_in['I2'])
    plt.plot(np.linspace(0,1,len(phi2U)),phi2U,'r')
    plt.title(f"correlation ratio:{corr_ratio_2U}")
    plt.xlabel('I2')
    plt.ylabel('U')

    plt.subplot(2,3,6)
    plt.scatter(all_in['U'],all_in['I3'])
    plt.plot(np.linspace(0,1,len(phi3U)),phi3U,'r')
    plt.title(f"correlation ratio:{corr_ratio_3U}")
    plt.xlabel('I3')
    plt.ylabel('U')

    """

    """
    plt.figure()
    plt.subplot(2,3,1)
    plt.scatter(map_data[0].test_bmus,map_data[0].test_u)
    plt.plot(np.linspace(0,1,len(phiB1U)),phiB1U,'r')
    plt.title(f"correlation ratio:{corr_ratio_B1U}")
    plt.xlabel('I1')
    plt.ylabel('I2')

    plt.subplot(2,3,2)
    plt.scatter(map_data[1].test_bmus,map_data[1].test_u)
    plt.plot(np.linspace(0,1,len(phiB2U)),phiB2U,'r')
    plt.title(f"correlation ratio:{corr_ratio_B2U}")
    plt.xlabel('I2')
    plt.ylabel('I3')

    plt.subplot(2,3,3)
    plt.scatter(map_data[2].test_bmus,map_data[2].test_u)
    plt.plot(np.linspace(0,1,len(phiB3U)),phiB3U,'r')
    plt.title(f"correlation ratio:{corr_ratio_B3U}")
    plt.xlabel('I1')
    plt.ylabel('I3')

    """



    plt.show()
