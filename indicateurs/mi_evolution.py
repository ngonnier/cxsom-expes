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
nxp = 10
stepf = 200
nf = 10000
NU=50
NP=500
plt.rcParams.update({
    'font.size': 16,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

INPUT_NAME = 'input'
IN_TEST_NAME = "input-test"
XP = 'cercle'

plt.rcParams['font.size'] = '16'
test_times = list(range(0,200,5))+list(range(200,10000,200))
#test_times = list(range(0,200,5))
print(test_times)
#calcul de im, retourne l'entropie et l'information mutuelle
def mutual_information(bmu_pos,U,n_u,n_p):
    nb_val = len(U)
    print(len(bmu_pos))
    hist_up,_,_ = np.histogram2d(bmu_pos,U,bins = [n_p,n_u])
    hist_u,_ = np.histogram(U,bins=n_u)
    hist_p,_ = np.histogram(bmu_pos,bins=n_p)
    eu = entropy(hist_u)
    ep = entropy(hist_p)
    hist_up = hist_up.flatten()
    epu = entropy(hist_up)
    im = eu + ep - epu
    return im,eu,ep

#calcul version kraskov
def mutual_information_kraskov(bmu_pos,U,n):
    bmu_pos = np.array(bmu_pos).reshape(-1,1)
    mi = mutual_info_regression(bmu_pos,U, n_neighbors=n)
    U1 = np.array(U).reshape(-1,1)
    eu = mutual_info_regression(U1,U, n_neighbors=n)
    return mi,eu

def mutual_information_conj(pos1,pos2,U,n_u,n_p):
    up = np.stack((pos1,pos2,U),axis=-1)
    hist_upp,edges = np.histogramdd(up,bins = [n_p,n_p,n_u])
    hist_u,_ = np.histogram(U,bins=n_u)
    hist_pp,_,_ = np.histogram2d(pos1,pos2,bins=n_p)
    hist_pp = hist_pp.flatten()
    hist_upp = hist_upp.flatten()
    eu = entropy(hist_u)
    epu = entropy(hist_upp)
    ep = entropy(hist_pp)
    im = eu + ep - epu
    return im,eu

#tableau de l'information mutuelle
#attention à garder les memes paramètres pour l'estimation >< -> si comparaison il faut, regenerer les 2.
def get_table(all_in,all_bmus):
    bigdict = {**all_in , **all_bmus}
    n = len(list(bigdict.keys()))
    info = np.zeros((n,n))
    for i,keys_i in enumerate(bigdict.keys()):
        for j,keys_b in enumerate(bigdict.keys()):
            im,eu = mutual_information_kraskov(bigdict[keys_b],bigdict[keys_i],10)
            info[i,j] = im/eu

    dat = pd.DataFrame(info,index = list(bigdict.keys()),columns = list(bigdict.keys()))
    print(dat)


#calcul du correlation_ratio

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

def difference_function(U,P,n_u,n_p):
    cr,phiY,phi_edges = correlation_ratio(U,P,n_u,n_p)
    #linear interpolation phiY
    interp = np.interp(P,phi_edges, phiY)
    diffs = np.zeros(len(U))
    for idx,elt in enumerate(U):
        diffs[idx] = (elt - interp[idx])**2
    mse = np.sqrt(1/len(diffs)*sum(diffs))
    return diffs,mse

def plot_diff_evolution(dir):
    nfilm = len(test_times)
    difftab1 = np.zeros((nxp,nfilm))
    difftab2 = np.zeros((nxp,nfilm))
    diffnctab1 = np.zeros((nxp,nfilm))
    diffnctab2 = np.zeros((nxp,nfilm))
    for xp in range(nxp):
        print(xp)
        path = os.path.join(dir,f'cercle{xp}')
        path_nc = os.path.join(dir,f'cercle_nc_{xp}')
        #film range
        all_in = get_inputs(path)
        all_in_nc = get_inputs(path_nc)
        for i,j in enumerate(test_times):
            all_bmus = get_bmus(path,j,map_names)
            all_bmus_nc = get_bmus(path_nc,j,map_names)
            d1tab,d1 = difference_function(all_in['U'],all_bmus['M1'],NU,NP)
            d2tab,d2= difference_function(all_in['U'],all_bmus['M2'],NU,NP)
            difftab1[xp,i] = d1
            difftab2[xp,i] = d2
            d1tabnc,d1nc= difference_function(all_in_nc['U'],all_bmus_nc['M1'],NU,NP)
            d2tabnc,d2nc = difference_function(all_in_nc['U'],all_bmus_nc['M2'],NU,NP)
            diffnctab1[xp,i] = d1nc
            diffnctab2[xp,i] = d2nc
    d_moy1 = np.mean(difftab1,axis = 0)
    d_std1 = np.std(difftab1,axis = 0)
    d_med1 = np.median(difftab1,axis=0)
    d_moy2= np.mean(difftab2,axis = 0)
    d_med2 = np.median(difftab2,axis=0)
    d_std2 = np.std(difftab2,axis = 0)
    d_moy1nc = np.mean(diffnctab1,axis = 0)
    d_std1nc = np.std(diffnctab1,axis = 0)
    d_med1nc = np.median(diffnctab1,axis=0)
    d_moy2nc = np.mean(diffnctab2,axis = 0)
    d_std2nc = np.std(diffnctab2,axis = 0)
    d_med2nc = np.median(diffnctab2,axis=0)
    #plot ces éléments
    fig,ax = plt.subplots(2,1)
    ax[0].set_title('Map X')
    ax[0].set_xlabel('timestep')
    ax[0].plot(test_times,d_moy1,'r',label='Moyenne, CxSOM')
    ax[0].fill_between(test_times,d_moy1-d_std1,d_moy1+d_std1,color='r',alpha=0.5)
    ax[0].plot(test_times,d_med1,'r--',label='Mediane,CxSOM')
    ax[0].plot(test_times,d_med1nc,'k--',label='Mediane, Carte Simple')
    ax[0].plot(test_times,d_moy1nc,'k',label='Moyenne, carte simple')
    ax[0].fill_between(test_times,d_moy1nc-d_std1nc,d_moy1nc+d_std1nc,color='k',alpha=0.5)
    ax[0].set_ylabel('$I_x$')
    ax[0].legend()
    ax[1].set_title('Map Y')
    ax[1].set_xlabel('timestep')
    ax[1].plot(test_times,d_med2,'r--',label='Mediane,CxSOM')
    ax[1].plot(test_times,d_med2nc,'k--',label='Mediane,Carte Simple')
    ax[1].plot(test_times,d_moy2,'r',label='Moyenne, CxSOM')
    ax[1].fill_between(test_times,d_moy2-d_std2,d_moy2+d_std2,color='r',alpha=0.5)
    ax[1].plot(test_times,d_moy2nc,'k',label='Moyenne, Carte simple')
    ax[1].fill_between(test_times,d_moy2nc-d_std2nc,d_moy2nc+d_std2nc,color='k',alpha=0.5)
    ax[1].set_ylabel('$I_x$')
    ax[1].legend()
    plt.show()



def plot_correlation_ratio(dir,xp,t):
    name = XP
    path = os.path.join(dir,f'{name}{xp}')
    path_nc = os.path.join(dir,f'{name}_nc_{xp}')
    np = 500
    nu = 500
    #film range
    all_in = get_inputs(path)
    all_in_nc = get_inputs(path_nc)
    nb_im = 0
    for test_time in t:
        all_bmus = get_bmus(path,test_time,map_names)
        all_bmus_nc = get_bmus(path_nc,test_time,map_names)
        cr1,phi1,p1_edges = correlation_ratio(all_in['U'],all_bmus['M1'],nu,np)
        cr2,phi2,p2_edges= correlation_ratio(all_in['U'],all_bmus['M2'],nu,np)
        crnc1,phinc1,p1_edgesnc = correlation_ratio(all_in_nc['U'],all_bmus_nc['M1'],nu,np)
        crnc2,phinc2,p2_edgesnc= correlation_ratio(all_in_nc['U'],all_bmus_nc['M2'],nu,np)

        d1tab,d1 = difference_function(all_in['U'],all_bmus['M1'],nu,np)
        d2tab,d2 = difference_function(all_in['U'],all_bmus['M2'],nu,np)
        d1nctab,d1nc= difference_function(all_in_nc['U'],all_bmus_nc['M1'],nu,np)
        d2nctab,d2nc = difference_function(all_in_nc['U'],all_bmus_nc['M2'],nu,np)

        fig,ax = plt.subplots(2,2)
        ax[0,0].scatter(all_bmus['M1'],all_in['U'],label='samples')
        ax[0,0].plot(p1_edges,phi1,'r',label='$\\mathbb{E}(U|\\Pi)$')
        ax[0,0].text(-0.2,1.2,'$\eta=%.3f$'%cr1)
        #ax[0,0].text(-0.2,1.3,'diff:%.3f'%d1)
        ax[1,0].scatter(all_bmus['M2'],all_in['U'])
        ax[1,0].plot(p2_edges,phi2,'r')
        ax[1,0].text(-0.2,-0.3,'$\eta=%.3f$'%cr2)
        #ax[1,0].text(-0.2,-0.2,'diff:%.3f'%d2)
        ax[0,0].set_title("Cartes CxSOM")
        ax[0,0].set_xlabel("$\\Pi^{(1)}$")
        ax[0,0].set_ylabel("$U$")
        ax[1,0].set_xlabel("$\\Pi^{(2)}$")
        ax[1,0].set_ylabel("$U$")
        ax[0,1].text(1.1,1.2,'$\eta=%.3f$'%crnc1)
        #ax[0,1].text(1.1,1.3,'diff:%.3f'%d1nc)
        ax[0,1].set_title("Cartes Simples")
        ax[0,1].set_xlabel("$\\Pi^{(1)}$")
        ax[0,1].set_ylabel("$U$")
        ax[1,1].set_xlabel("$\\Pi^{(2)}$")
        ax[1,1].set_ylabel("$U$")
        ax[1,1].text(1.1,-0.3,'$\eta=%.3f$'%crnc2)
        #ax[1,1].text(1.1,-0.15,'diff:%.3f'%d2nc)
        ax[0,1].scatter(all_bmus_nc['M1'],all_in_nc['U'])
        ax[0,1].plot(p1_edgesnc,phinc1,'r')
        ax[1,1].scatter(all_bmus_nc['M2'],all_in_nc['U'])
        ax[1,1].plot(p2_edgesnc,phinc2,'r')

        ax[0,0].legend(loc='upper center',bbox_to_anchor=(0.5,1.05),ncol=2)
        plt.savefig(f'correlation_ratio_xp_{xp}_t_{nb_im:03d}')
        plt.close()
        nb_im+=1

def get_inputs(path):
    #get inputs
    in_path = os.path.join(path,IN_TEST_NAME)
    files = [f for f in os.listdir(in_path)]
    all_in = dict()
    for f in files:
        var_path = os.path.join(in_path,f)
        with cx.variable.Realize(var_path) as inp:
            input = [inp[i] for i in range(inp.time_range()[1])]
            var_name = f[:-4]
            all_in[var_name]=input

    return all_in
    #keys  = list(all_in.keys())

def get_bmus(path,j,map_names):
    #get bmus
    all_bmus = dict()
    #get BMUs
    if(XP=='cercle'):
        test_name = f'{j}'
    else:
        test_name = f'{j:04d}'
    for map_name in map_names:
        if(INPUT_NAME=='in'):
            path_bmu = os.path.join(path,"zfrz-"+test_name+"-out",map_name,'BMU.var')
        else:
            path_bmu = os.path.join(path,"rlx-test-"+test_name,map_name,'BMU.var')

        with cx.variable.Realize(path_bmu) as fbmu:
            r = fbmu.time_range()
            bmus = [fbmu[i] for i in range(r[1])]
            all_bmus[map_name] = np.array(bmus)

    return all_bmus

def plot_mi_comparison(dir,map_names):
    nfilm = len(test_times)
    mitab_k1 = np.zeros((nxp,nfilm))
    mitab_k2 = np.zeros((nxp,nfilm))
    minctab_k1 = np.zeros((nxp,nfilm))
    minctab_k2 = np.zeros((nxp,nfilm))

    xu1_tab = np.zeros((nxp,nfilm))
    pix1_tab = np.zeros((nxp,nfilm))
    xunc1_tab = np.zeros((nxp,nfilm))
    pixnc1_tab = np.zeros((nxp,nfilm))

    xu2_tab = np.zeros((nxp,nfilm))
    pix2_tab = np.zeros((nxp,nfilm))
    xunc2_tab = np.zeros((nxp,nfilm))
    pixnc2_tab = np.zeros((nxp,nfilm))

    im_conj_u = np.zeros((nxp,nfilm))
    im_nc_conj_u = np.zeros((nxp,nfilm))
    imx_tab=np.zeros(nxp)
    imy_tab=np.zeros(nxp)

    for xp in range(nxp):
        print(xp)
        path = os.path.join(dir,f'{XP}{xp}')
        path_nc = os.path.join(dir,f'{XP}_nc_{xp}')
        #film range
        all_in = get_inputs(path)
        all_in_nc = get_inputs(path_nc)
        bins = 500
        x_n = list(map(lambda xx:round(xx*bins)/bins, all_in['I1']))
        y_n = list(map(lambda xx:round(xx*bins)/bins, all_in['I2']))
        imx,eux,_ = mutual_information(x_n,all_in['I1'],NU,NP)
        imy,euy,_ = mutual_information(y_n,all_in['I2'],NU,NP)
        imxn,euxn,_ = mutual_information(x_n,all_in['U'],NU,NP)
        imyn,euyn,_ = mutual_information(y_n,all_in['U'],NU,NP)

        imx_tab[xp] = imx/euxn
        imy_tab[xp] = imy/euyn

        for i,j in enumerate(test_times):
            all_bmus = get_bmus(path,j,map_names)
            all_bmus_nc = get_bmus(path_nc,j,map_names)
            #values
            imk1,euk1,_= mutual_information(all_bmus['M1'],all_in['U'],NU,NP)
            imk2,euk2,_= mutual_information(all_bmus['M2'],all_in['U'],NU,NP)
            x1u,eu1,_ = mutual_information(all_in['I1'],all_in['U'],NU,NP)
            pix1,ex1,_ = mutual_information(all_bmus['M1'],all_in['I1'],NU,NP)
            x2u,eu2,_ = mutual_information(all_in['I2'],all_in['U'],NU,NP)
            pix2,ex2,_ = mutual_information(all_bmus['M2'],all_in['I2'],NU,NP)
            imconj,euconj = mutual_information_conj(all_bmus['M1'],all_bmus['M2'],all_in['U'],NU,NP)


            mitab_k1[xp,i] = imk1/euk1
            mitab_k2[xp,i] = imk2/euk2
            xu1_tab[xp,i] = x1u/eu1
            xu2_tab[xp,i] = x2u/eu2
            pix1_tab[xp,i] = pix1/ex1
            pix2_tab[xp,i] = pix2/ex2
            im_conj_u[xp,i] = imconj/euconj

            imnck1,eunck1,_= mutual_information(all_bmus_nc['M1'],all_in_nc['U'],NU,NP)
            imnck2,eunck2,_= mutual_information(all_bmus_nc['M2'],all_in_nc['U'],NU,NP)
            x1ncu,eunc1,_ = mutual_information(all_in_nc['I1'],all_in_nc['U'],NU,NP)
            pixnc1,exnc1,_ = mutual_information(all_bmus_nc['M1'],all_in_nc['I1'],NU,NP)
            x2ncu,eunc2,_ = mutual_information(all_in_nc['I2'],all_in_nc['U'],NU,NP)
            pixnc2,exnc2,_ = mutual_information(all_bmus_nc['M2'],all_in_nc['I2'],NU,NP)
            imconjnc,euconjnc = mutual_information_conj(all_bmus_nc['M1'],all_bmus_nc['M2'],all_in_nc['U'],NU,NP)
            minctab_k1[xp,i] = imnck1/eunck1
            minctab_k2[xp,i] = imnck2/eunck2
            xunc1_tab[xp,i] = x1ncu/eunc1
            xunc2_tab[xp,i] = x2ncu/eunc2
            pixnc1_tab[xp,i] = pixnc1/exnc1
            pixnc2_tab[xp,i] = pixnc2/exnc2
            im_nc_conj_u[xp,i] = imconjnc/euconjnc

    #tracer moyenne et ecart type sur toutes les xp =moyenne par colonne
    mik_moy1 =  np.mean(mitab_k1,axis = 0)
    mik_std1 =  np.std(mitab_k1,axis = 0)
    mik_moy2 =  np.mean(mitab_k2,axis = 0)
    mik_std2 =  np.std(mitab_k2,axis = 0)

    minck_moy1 =  np.mean(minctab_k1,axis = 0)
    minck_std1 =  np.std(minctab_k1,axis = 0)
    minck_moy2 =  np.mean(minctab_k2,axis = 0)
    minck_std2 =  np.std(minctab_k2,axis = 0)

    xu1_moy1 =  np.mean(xu1_tab,axis = 0)
    xu1_std1 =  np.std(xu1_tab,axis = 0)
    xu2_moy =  np.mean(xu2_tab,axis = 0)
    xu2_std =  np.std(xu2_tab,axis = 0)
    pix1_moy = np.mean(pix1_tab,axis = 0)
    pix1_std = np.std(pix1_tab,axis = 0)
    pix2_moy = np.mean(pix2_tab,axis = 0)
    pix2_std = np.std(pix2_tab,axis = 0)

    pixnc1_moy = np.mean(pixnc1_tab,axis = 0)
    pixnc2_moy = np.mean(pixnc2_tab,axis = 0)

    im_conj_u_moy = np.mean(im_conj_u,axis=0)
    im_nc_conj_u_moy = np.mean(im_nc_conj_u,axis=0)
    imx_moy = np.mean(imx_tab)
    imy_moy = np.mean(imy_tab)


    #plot ces éléments
    fig,ax = plt.subplots(2,1)
    ax[0].set_title('Map X')
    ax[0].set_xlabel('timestep')
    ax[0].plot(test_times,mik_moy1,'r',label='$I_x(U|\\Pi^{(1)})$, CxSOM')
    #ax[0].fill_between(test_times,mik_moy1-mik_std1,mik_moy1+mik_std1,color='r',alpha=0.5)
    ax[0].plot(test_times,minck_moy1,'orange',label='$I_x(U|\\Pi^{(1)})$, Carte simple')
    ax[0].plot(test_times,pixnc1_moy,'k',label='$I_x(X^{(1)}|\\Pi^{(1)})$, Carte simple')
    #ax[0].fill_between(test_times,minck_moy1-minck_std1,minck_moy1+minck_std1,color='k',alpha=0.5)
    ax[0].plot(test_times,pix1_moy,'b',label='$I_x(X^{(1)}|\\Pi^{(1)})$, CxSOM')
    ax[0].plot(test_times,xu1_moy1,'g',label='$I_x(U|X^{(1)})$')
    ax[0].plot([0,test_times[-1]],[imx_moy,imx_moy],'k--',label='objectif théorique $I_x(X^{(1)}|\\Pi^{(1)})$')
    ax[0].plot([0,test_times[-1]],[imxn/euxn,imxn/euxn],'g--',label='objectif théorique $I_x(U|\\Pi)$')
    ax[0].plot(test_times,im_conj_u_moy,'m',label='$I_x(U|(\\Pi^{(1)},\\Pi^{(2)})$, CxSOM')
    ax[0].plot(test_times,im_nc_conj_u_moy,'m--',label='$I_x(U|(\\Pi^{(1)},\\Pi^{(2)})$, carte simple')
    #ax[0].fill_between(test_times,minck_moy1-minck_std1,minck_moy1+minck_std1,color='k',alpha=0.5)
    ax[0].set_ylabel('$I_x$')
    ax[0].legend()

    ax[1].set_title('Map Y')
    ax[1].set_xlabel('timestep')
    ax[1].plot(test_times,mik_moy2,'r',label='$I_x(U|\\Pi^{(2)})$, CxSOM')
    #ax[1].fill_between(test_times,mik_moy2-mik_std2,mik_moy2+mik_std2,color='r',alpha=0.5)
    ax[1].plot(test_times,minck_moy2,'orange',label='$I_x(U|\\Pi^{(2)})$, Carte simple')
    ax[1].plot(test_times,pixnc2_moy,'k',label='$I_x(X^{(2)}|\\Pi^{(1)})$, Carte simple')
    #ax[1].fill_between(test_times,minck_moy2-minck_std2,minck_moy2+minck_std2,color='k',alpha=0.5)
    ax[1].plot(test_times,pix2_moy,'b',label='$I_x(X^{(2)}|\\Pi^{(2)})$, CxSOM')
    ax[1].plot(test_times,xu2_moy,'g',label='$I_x(U|X^{(2)})$')
    ax[1].plot([0,test_times[-1]],[imy_moy,imy_moy],'k--',label='objectif théorique $I_x(X^{(2)}|\\Pi^{(2)})$')
    ax[1].plot([0,test_times[-1]],[imyn/euyn,imyn/euyn],'g--',label='objectif théorique $I_x(U|\\Pi)$')
    ax[1].set_ylabel('$I_x$')
    ax[1].legend()

    plt.show()

#evolution films plot
def plot_mi_evol(dir,map_names):
    nfilm = len(test_times)
    # mitab1 = np.zeros((nxp,nfilm))
    # mitab2 = np.zeros((nxp,nfilm))
    crtab1 = np.zeros((nxp,nfilm))
    crtab2 = np.zeros((nxp,nfilm))
    #mitab_k1 = np.zeros((nxp,nfilm))
    #mitab_k2 = np.zeros((nxp,nfilm))

    # minctab1 = np.zeros((nxp,nfilm))
    # minctab2 = np.zeros((nxp,nfilm))
    crnctab1 = np.zeros((nxp,nfilm))
    crnctab2 = np.zeros((nxp,nfilm))
    #minctab_k1 = np.zeros((nxp,nfilm))
    #minctab_k2 = np.zeros((nxp,nfilm))



    #mitab11 = np.zeros((nxp,nfilm))
    #minctab_k11 = np.zeros((nxp,nfilm))

    for xp in range(nxp):
        print(xp)
        path = os.path.join(dir,f'{XP}{xp}')
        path_nc = os.path.join(dir,f'{XP}_nc_{xp}')
        #film range
        all_in = get_inputs(path)
        all_in_nc = get_inputs(path_nc)
        for i,j in enumerate(test_times):
            all_bmus = get_bmus(path,j,map_names)
            all_bmus_nc = get_bmus(path_nc,j,map_names)
            #values

            #im1,eu1,_ = mutual_information(all_bmus['M1'],all_in['U'],NU,NP)
            #im2,eu2,_ = mutual_information(all_bmus['M2'],all_in['U'],NU,NP)
            #imk1,euk1= mutual_information_kraskov(all_bmus['M1'],all_in['U'],10)
            #imk2,euk2= mutual_information_kraskov(all_bmus['M2'],all_in['U'],10)
            #x1u,eu1 = mutual_information_kraskov(all_in['I1'],all_in['U'],10)
            #pix1,ex1 = mutual_information_kraskov(all_bmus['M1'],all_in['I1'],10)

            cr1,phi1,p1_edges = correlation_ratio(all_in['U'],all_bmus['M1'],NP)
            cr2,phi2,p2_edges= correlation_ratio(all_in['U'],all_bmus['M2'],NP)

            """
            im1,eu1,_ = mutual_information(all_bmus['M1'],all_in['U'],50,500)
            imk1,euk1,_= mutual_information(all_bmus['M1'],all_in['U'],100,500)
            im2,eu2,_ = mutual_information(all_bmus['M2'],all_in['U'],50,100)
            imk2,euk2= mutual_information_kraskov(all_bmus['M2'],all_in['U'],10)
            cr1,phi1 = correlation_ratio(all_in['U'],all_bmus['M1'],50,100)
            cr2,phi2 = correlation_ratio(all_in['U'],all_bmus['M2'],50,100)
            """
            #mitab1[xp,i] = im1/eu1
            #mitab_k1[xp,i] = imk1/euk1

            crtab1[xp,i] = cr1
            #mitab2[xp,i] = im2/eu2
            #mitab_k2[xp,i] = imk2/euk2
            crtab2[xp,i] = cr2

            #imk11,euk11,_= mutual_information(all_bmus_nc['M1'],all_in_nc['U'],NU,NP)
            #mitab11[xp,i] = imk11/euk11


            #imnc1,eunc1,_ = mutual_information(all_bmus_nc['M1'],all_in_nc['U'],NU,NP)
            #imnck1,eunck1= mutual_information_kraskov(all_bmus_nc['M1'],all_in_nc['U'],10)
            #imnc2,eunc2,_ = mutual_information(all_bmus_nc['M2'],all_in_nc['U'],NU,NP)
            #imnck2,eunck2= mutual_information_kraskov(all_bmus_nc['M2'],all_in_nc['U'],10)

            crnc1,phinc1,_ = correlation_ratio(all_in_nc['U'],all_bmus_nc['M1'],NP)
            crnc2,phinc2,_ = correlation_ratio(all_in_nc['U'],all_bmus_nc['M2'],NP)

            """
            imnc1,eunc1,_ = mutual_information(all_bmus_nc['M1'],all_in_nc['U'],50,500)
            imnck1,eunck1,_= mutual_information(all_bmus_nc['M1'],all_in_nc['U'],100,500)
            #imnck11,eunck11,_= mutual_information(all_bmus_nc['M1'],all_in_nc['U'],NU,NP)
            imnc2,eunc2,_ = mutual_information(all_bmus_nc['M2'],all_in_nc['U'],50,500)
            imnck2,eunck2,_= mutual_information(all_bmus_nc['M2'],all_in_nc['U'],100,500)
            #imnck22,eunck22,_= mutual_information(all_bmus_nc['M2'],all_in_nc['U'],NU,NP)
            crnc1,phinc1 = correlation_ratio(all_in_nc['U'],all_bmus_nc['M1'],50,100)
            crnc2,phinc2 = correlation_ratio(all_in_nc['U'],all_bmus_nc['M2'],50,100)
            """
            #minctab1[xp,i] = imnc1/eunc1
            #minctab_k1[xp,i] = imnck1/eunck1

            crnctab1[xp,i] = crnc1
            #minctab2[xp,i] = imnc2/eunc2
            #minctab_k2[xp,i] = imnck2/eunck2
            crnctab2[xp,i] = crnc2

            #minctab_k11[xp,i] = imnck11/eunck11

    #tracer moyenne et ecart type sur toutes les xp =moyenne par colonne

    #mi_moy1 = np.mean(mitab1,axis = 0)
    #mi_std1 = np.std(mitab1, axis = 0)
    #mik_moy1 =  np.mean(mitab_k1,axis = 0)
    #mik_std1 =  np.std(mitab_k1,axis = 0)

    cr_moy1 = np.mean(crtab1,axis = 0)
    cr_std1 = np.std(crtab1,axis = 0)

    #mik_moy11 =  np.mean(mitab11,axis = 0)
    #mik_std11 =  np.std(mitab11,axis =0)
    #minck_moy11 =  np.mean(minctab_k11,axis = 0)
    #minck_std11 =  np.std(minctab_k11,axis =0)


    # mi_moy2 = np.mean(mitab2,axis = 0)
    # mi_std2 = np.std(mitab2, axis = 0)
    #mik_moy2 =  np.mean(mitab_k2,axis = 0)
    #mik_std2 =  np.std(mitab_k2,axis = 0)

    cr_moy2= np.mean(crtab2,axis = 0)
    cr_std2 = np.std(crtab2,axis = 0)


    # minc_moy1 = np.mean(minctab1,axis = 0)
    # minc_std1 = np.std(minctab1, axis = 0)
    #minck_moy1 =  np.mean(minctab_k1,axis = 0)
    #minck_std1 =  np.std(minctab_k1,axis = 0)

    crnc_moy1 = np.mean(crnctab1,axis = 0)
    crnc_std1 = np.std(crnctab1,axis = 0)


    # minc_moy2 = np.mean(minctab2,axis = 0)
    # minc_std2 = np.std(minctab2, axis = 0)
    #minck_moy2 =  np.mean(minctab_k2,axis = 0)
    #minck_std2 =  np.std(minctab_k2,axis = 0)

    crnc_moy2= np.mean(crnctab2,axis = 0)
    crnc_std2 = np.std(crnctab2,axis = 0)

    # #plot ces éléments
    # fig,ax = plt.subplots(2,1)
    # ax[0].set_title('Carte $M^{(1)}$')
    # # ax[0].plot(test_times,mi_moy1,'b',label='CxSOM')
    # # ax[0].fill_between(test_times,mi_moy1-mi_std1,mi_moy1+mi_std1,color='b',alpha=0.5)
    # ax[0].set_xlabel('timestep')
    # ax[0].plot(test_times,mik_moy1,'r',label='$I_x(U|\\Pi^{(1)})$, CxSOM')
    # ax[0].fill_between(test_times,mik_moy1-mik_std1,mik_moy1+mik_std1,color='r',alpha=0.5)
    #
    # #ax[0].plot(test_times,mik_moy11,'r',label='UC, binning 500')
    # #ax[0].fill_between(test_times,mik_moy11-mik_std11,mik_moy11+mik_std11,color='pink',alpha=0.5)
    #
    # ax[0].plot(test_times,minc_moy1,'g',label='Carte simple')
    # ax[0].fill_between(test_times,minc_moy1-minc_std1,minc_moy1+minc_std1,color='g',alpha=0.5)
    # #ax[0].plot(test_times,minck_moy1,'k',label='$I_x(U|\\Pi^{(1)})$, Carte simple')
    # #ax[0].fill_between(test_times,minck_moy1-minck_std1,minck_moy1+minck_std1,color='k',alpha=0.5)
    # ax[0].set_ylabel('$I_x$')
    # #ax[0].plot(test_times,minck_moy11,'k',label='UC non connecté, binning 500')
    # #ax[0].fill_between(test_times,minck_moy11-minck_std11,minck_moy11+minck_std11,color='darkblue',alpha=0.5)
    #
    # ax[0].legend()
    #
    # ax[1].set_title('Carte $M^{(2)}$')
    # ax[1].plot(test_times,mi_moy2,'b', label = 'CxSOM')
    # ax[1].fill_between(test_times,mi_moy2-mi_std2,mi_moy2+mi_std2,color='b',alpha=0.5)
    # ax[1].set_xlabel('timestep')
    # #ax[1].plot(test_times,mik_moy2,'r',label='$I_x(U|\\Pi^{(2)})$, CxSOM')
    # #ax[1].fill_between(test_times,mik_moy2-mik_std2,mik_moy2+mik_std2,color='r',alpha=0.5)
    # ax[1].plot(test_times,minc_moy2,'g',label='Carte simple')
    # ax[1].fill_between(test_times,minc_moy2-minc_std2,minc_moy2+minc_std2,color='g',alpha=0.5)
    # #ax[1].plot(test_times,minck_moy2,'k',label='$I_x(U|\\Pi^{(2)})$, Carte simple')
    # #ax[1].fill_between(test_times,minck_moy2-minck_std2,minck_moy2+minck_std2,color='k',alpha=0.5)
    # ax[1].set_ylabel('$I(U,\Pi^{(2)})$')
    # ax[1].legend()

    fig2,ax2 = plt.subplots(2,1)
    ax2[0].plot(test_times,cr_moy1,'b',label='$\eta(U|\Pi^{(1)})$,CxSOM')
    ax2[0].fill_between(test_times,cr_moy1-cr_std1,cr_moy1+cr_std1,color='b',alpha=0.5)

    ax2[0].plot(test_times,crnc_moy1,'g',label='$\eta(U|\Pi^{(1)})$,carte simple')
    ax2[0].fill_between(test_times,crnc_moy1-crnc_std1,crnc_moy1+crnc_std1,color='g',alpha=0.5)

    ax2[1].plot(test_times,cr_moy2,'r',label = '$\eta(U|\Pi^{(2)})$,CxSOM')

    ax2[1].fill_between(test_times,cr_moy2-cr_std2,cr_moy2+cr_std2,color='r',alpha=0.5)

    ax2[1].plot(test_times,crnc_moy2,'k',label='$\eta(U|\Pi^{(2)})$,carte simple')
    ax2[1].fill_between(test_times,crnc_moy2-crnc_std2,crnc_moy2+crnc_std2,color='k',alpha=0.5)

    ax2[0].legend()
    ax2[1].legend()


    ax2[0].set_ylim([0, 1])
    ax2[1].set_ylim([0, 1])
    
    plt.show()

def plot_estimation(dir,map_names):
    test_time = 9800
    estimation_parameters= [10,20,50,100,200,500,1000,1200,1500,2000]
    estimation_parameters_k = range(3,len(estimation_parameters)+3)
    mitab = np.zeros((nxp,len(estimation_parameters)))
    minctab = np.zeros((nxp,len(estimation_parameters)))
    miktab = np.zeros((nxp,len(estimation_parameters)))
    mincktab = np.zeros((nxp,len(estimation_parameters)))
    for xp in range(nxp):
        print(xp)
        path = os.path.join(dir,f'{XP}{xp}')
        path_nc = os.path.join(dir,f'{XP}_nc_{xp}')
        #film range
        all_in = get_inputs(path)
        all_in_nc = get_inputs(path_nc)
        all_bmus = get_bmus(path,test_time,map_names)
        all_bmus_nc = get_bmus(path_nc,test_time,map_names)
        for i,u_parameter in enumerate(estimation_parameters):
            im1,eu1,_ = mutual_information(all_bmus['M1'],all_in['U'],u_parameter,500)
            imnc1,eunc1,_ = mutual_information(all_bmus_nc['M1'],all_in_nc['U'],u_parameter,500)
            imk1,euk1= mutual_information_kraskov(all_bmus['M1'],all_in['U'],estimation_parameters_k[i])
            imnck1,eunck1 = mutual_information_kraskov(all_bmus_nc['M1'],all_in_nc['U'],estimation_parameters_k[i])
            mitab[xp,i] = im1/eu1
            minctab[xp,i] = imnc1/eunc1
            miktab[xp,i] = imk1/euk1
            mincktab[xp,i] = imnck1/eunck1

    mi_moy1 = np.mean(mitab,axis = 0)
    mi_std1 = np.std(mitab, axis = 0)
    minc_moy1 = np.mean(minctab,axis = 0)
    minc_std1 = np.std(minctab, axis = 0)
    mik_moy1 = np.mean(mitab,axis = 0)
    mik_std1 = np.std(mitab, axis = 0)
    minck_moy1 = np.mean(minctab,axis = 0)
    minck_std1 = np.std(minctab, axis = 0)
    fig,ax = plt.subplots(2,1)
    #plt.plot(estimation_parameters,mi_moy1,color='b',label='CxSOM')
    ax[0].errorbar(estimation_parameters,mi_moy1,yerr=mi_std1,fmt='-o',label='CXSOM')
    ax[0].errorbar(estimation_parameters,minc_moy1,yerr=minc_std1,fmt='-o',label='Simple')
    ax[0].set_title('Valeur de I(U,BMU) après apprentissage, estimation histogramme en fonction des paramètres')
    ax[0].set_xlabel('Nombre de boites pour U')
    ax[0].set_ylabel('Valeur de l\'information mutuelle normalisée')
    ax[1].set_title('Valeur de I(U,BMU) après apprentissage, estimation Kraskov, en fonction des paramètres')
    ax[1].errorbar(estimation_parameters_k,mik_moy1,yerr=mik_std1,fmt='-o',label='CXSOM')
    ax[1].errorbar(estimation_parameters_k,minck_moy1,yerr=minck_std1,fmt='-o',label='Simple')
    ax[1].set_xlabel('Nombre de voisins pour kraskov')
    ax[1].set_ylabel('Valeur de l\'information mutuelle normalisée')
    plt.legend()
    plt.show()

def mi_test():
    bins = 500
    xp=0
    path = os.path.join(dir,f'cercle{xp}')
    #film range
    all_in = get_inputs(path)
    x_n = list(map(lambda xx:round(xx*bins)/bins, all_in['I1']))
    print(x_n)
    imx,eux = mutual_information_kraskov(x_n,all_in['I1'],10)
    uc = imx/eux
    plt.figure()
    plt.scatter(x_n,all_in['I1'])
    plt.title(f'Discrétisation: {bins}, coefficient: {uc}')
    plt.show()



if __name__ == '__main__':
    dir = "xp_cercle"
    map_names = ["M1","M2"]
    plot_mi_evol(dir, map_names)
    #plot_mi_comparison(dir,map_names)
    #plot_estimation(dir,map_names)
    #plot_diff_evolution(dir)
    #plot_correlation_ratio(dir,0,test_times)
    #mi_test()
