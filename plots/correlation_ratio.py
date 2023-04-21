import os
import time
import numpy as np
import pycxsom as cx
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import sys
import struct
from scipy.stats import entropy
#from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde


def correlation_ratio(U,P,n_p):
    phiY = np.zeros(n_p)
    #hist_p,pedge = np.histogram(P,bins = n_p)
    #hist_p = np.sum(hist_up, axis = 1)
    triu = np.empty(n_p,dtype=object)
    moyenne = np.zeros(n_p)
    histo = np.zeros(n_p)
    var =  np.zeros(n_p)
    for i in range(n_p):
        triu[i] = []

    for i in range(len(U)):
        pos = P[i]
        u = U[i]
        ipos0 = int((n_p-1)*pos)
        histo[ipos0] = histo[ipos0] + 1
        moyenne[ipos0] +=u
        triu[ipos0].append(u)

    for i in range(n_p):
        if histo[i] == 0:
            phiY[i] = -1
            var[i] = -1
        else:
            #phiY[i] = 1/hist_p[i] * sum([uedge[j]*hist_up[i,j] for j in range(n_u)])
            phiY[i] = 1/histo[i] * moyenne[i]
            var[i] = 1/histo[i] * sum([(u -  phiY[i])**2 for u in triu[i]])

        mask= phiY > -1.
        vup = 0
        n=0
        for i in range(n_p):
            if(var[i]>-1.):
                vup += var[i]
                n+=1
        Vup = (1/n)*vup
        moyu = 1/len(U) * np.sum(U)
        vu = 0
        for u in U:
            vu += (u - moyu)**2
        Vu = (1/len(U)) * vu
        cr = 1 - Vup/Vu

    #cr = np.var(phiY)/np.var(U)
    phiY= phiY.reshape((n_p,))
    mask= phiY > -1.
    phiY = phiY[mask]
    p = np.linspace(0,1,n_p)
    phi_edges = p[mask]
    #cr = np.sqrt(np.var(phiY)/np.var(U))
    return np.sqrt(cr),phiY, phi_edges

def l2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def correlation_ratio_2D(U,P,n_p):
    histo = np.zeros((n_p,n_p))
    moyenne = np.zeros((n_p,n_p,2))
    phiY = np.zeros((n_p,n_p,2))
    triu = np.empty((n_p, n_p), dtype = object)
    for i in range(n_p):
        for j in range(n_p):
            triu[i,j] = []
    var = np.zeros((n_p,n_p))
    for i in range(len(U)):
        pos = P[i]
        u = U[i]
        ipos0 = int((n_p-1)*(pos[0]))
        ipos1 = int((n_p-1)*(pos[1]))
        histo[ipos0,ipos1] = histo[ipos0,ipos1] + 1
        moyenne[ipos0,ipos1,:] += u
        triu[ipos0,ipos1].append(u)

    for i in range(n_p):
        for j in range(n_p):
            if histo[i,j] == 0:
                phiY[i,j] = -1
                var[i,j] = -1
            else:
                phiY[i,j,:] = 1/histo[i,j] * moyenne[i,j]
                var[i,j] = 1/len(triu[i,j]) * sum([l2(u, phiY[i,j,:] ) for u in triu[i,j]])

    mask= phiY > -1.
    vup = 0
    n=0
    for i in range(n_p):
        for j in range(n_p):
            if(var[i,j]>-1.):
                vup += var[i,j]
                n+=1

    Vup = (1/n)*vup

    moyu = 1/len(U) * np.sum(U, axis = 0)
    vu = 0
    for u in U:
        vu += l2(u, moyu)
    Vu = (1/len(U)) * vu
    cr = 1 - Vup/Vu

    return np.sqrt(cr),phiY


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
    up[:,-u.shape[1]:] = u
    hist_upp,edges = np.histogramdd(up,bins = [n_p]*p.shape[1]+[n_u]*u.shape[1])
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

# def count_zones(U,P):
#
#
#     def aux_count(pos, tab):
#
#         #Max : on arrive a 500 zones
#
#         #Calcul de l'énergie droite et gauche
#
#         #si energie pas satisdaisante, on calcule le nb de zones sur les zones droites et gauche de P.
#         t1 = filter(lambda x: x > pos, tab)
#         t2 = filter(lambda x: x <= pos, tab)
#
#         n1,
#

def mi_continuous(U,P,n_P, n_U):
    #estimation non paramétrique des dp a l'aide d'histogrammes/mixtures gaussiennes et calcule l'information mutuelle
    # amélioriation possible : choisir les paramètres des histogrammes avec la méthode de découpage en arbres.
    # use grid search cross-validation to optimize the bandwidth
    params = {"bandwidth": np.logspace(-1, 1, 20)}
    # grid1 = GridSearchCV(KernelDensity(), params)
    # grid1.fit(U)
    # kdeU = grid1.best_estimator_
    #
    # grid2 = GridSearchCV(KernelDensity(), params)
    # grid2.fit(P)
    # kdeP = grid2.best_estimator_
    #
    # if(len(U.shape)==1):
    #     U = np.reshape(U, (-1,1))
    # if(len(P.shape)==1):
    # P = np.reshape(P, (-1,1))
    #kdeup = grid3.best_estimator_
    kernelU = gaussian_kde(U)
    kernelP = gaussian_kde(P)
    kernelUP = gaussian_kde(UP)








# use the best estimator to compute the kernel density estimate





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
