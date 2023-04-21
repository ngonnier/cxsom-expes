import sys
import matplotlib.pyplot as plt
import pycxsom as cx
import os
import sys
import numpy as np

def get_rlx(dir,map_name,tnum):
    path = os.path.join(dir,f'zfrz-{tnum:04d}-out','Cvg.var')
    with cx.variable.Realize(path) as frlx:
        r = frlx.time_range()
        rlx = np.zeros(r[1])
        for i in range(r[1]):
            rlx[i] = frlx[i]
    return rlx



if __name__ == '__main__':
    if len(sys.argv) < 5:
        sys.exit("Usage: python3 convergence_relax.py <path_to_dir> <prefix> <begin> <end> <step>")

    plt.rc('text', usetex=True)
    plt.rcParams['font.size'] = '16'
    #dirs = ['../../xp_list/2SOM_SU_000','../../xp_list/2SOM_SU_003','../../xp_list/2SOM_SU_004']
    dirs = ['../../xp_list/3SOM_rcre', '../../xp_list/3SOM_cercle']
    prefix = sys.argv[2]
    begin = int(sys.argv[3])
    end = int(sys.argv[4])
    step = int(sys.argv[5])

    #xps = range(begin,end,step)
    xps = list(range(0,1000,20))
    moy = np.zeros((len(xps),len(dirs)))
    taux = np.zeros((len(xps),len(dirs)))
    for j,expe in enumerate(dirs):
        for i,xx in enumerate(xps):
            path = os.path.join(expe,f'{prefix}-{xx:04d}-rlx','Cvg.var')
            with cx.variable.Realize(path) as frlx:
                r = frlx.time_range()
                cvg = [frlx[at] for at in range(r[0],r[1])]
            moy[i,j] = sum(cvg)/len(cvg)
            taux[i,j] = 100*len(list(filter(lambda x:x <200, cvg)))/len(cvg)


    fig,axes = plt.subplots(2,1)
    axes[0].plot(xps,moy[:,0],label='Sans zones')
    axes[0].plot(xps,moy[:,1],label='Avec zones')
    #axes[0].plot(xps,moy[:,2],label='$r_c = 0.02$')
    # axes[0].fill_between(xps,moy_moy+moy_std, moy_moy-moy_std,alpha=0.6)
    axes[1].plot(xps,taux[:,0],label='Sans zones')
    axes[1].plot(xps,taux[:,1],label='Avec zones')
    #axes[1].plot(xps,taux[:,2],label='$r_c = 0.02$')
    #axes[1].fill_between(xps,taux_moy+taux_std, taux_moy-taux_std,alpha=0.6)
    axes[0].set_xlabel('itération')
    axes[0].set_ylabel('nombre moyen de pas de relaxation')
    axes[1].set_xlabel('itération')
    axes[1].set_ylabel('taux de convergence, \%')


    # #calcul des valeurs pour expe 005 (begin step end different)
    #
    # begin = 1000
    # end = 500000
    # step = 5000
    # xps = range(begin,end,step)
    # dir = '../../xp_list/2SOM_S_003'
    # moy = np.zeros(len(xps))
    # taux = np.zeros(len(xps))
    # for i,xx in enumerate(xps):
    #     path = os.path.join(dir,f'{prefix}-{xx}-rlx','Cvg.var')
    #     with cx.variable.Realize(path) as frlx:
    #         r = frlx.time_range()
    #         cvg = [frlx[at] for at in range(r[0],r[1])]
    #     moy[i] = sum(cvg)/len(cvg)
    #     taux[i] = 100*len(list(filter(lambda x:x <1000, cvg)))/len(cvg)
    #
    # axes[0].plot(xps,moy,'k', label='$r_c = 0.05$')
    # axes[1].plot(xps,taux,'k', label='$r_c = 0.05$')
    axes[0].legend()
    axes[1].legend()



    plt.show()
