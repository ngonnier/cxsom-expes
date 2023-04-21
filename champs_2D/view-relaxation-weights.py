import pycxsom as cx
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def find_best_rowcol(n):
    min_r = n
    a = 0
    m = int(np.sqrt(n))
    for i in range(1,m+1):
        r = n % i
        q = n/i
        if(r<min_r):
            min_r = r
            a = i
    rows = i
    if r == 0:
        cols = int(n/i)
    else:
        cols = int(n/i) + 1
    return cols,rows


if __name__ == "__main__":

    if len(sys.argv)<4:
        sys.exit("usage: python3 plot_weights.py <data_dir> <timestep> <prefix> <ntraj> <time_inp>")
    dir= sys.argv[1]
    timestep = int(sys.argv[2])
    prefix = sys.argv[3]
    ntraj = int(sys.argv[4])
    timeinp = int(sys.argv[5])
    path = os.path.join(dir, 'wgt')
    maps = os.listdir(path)
    inp_path = os.path.join(dir,'ztest-in')
        #inputs = os.listdir(inp_path)


    def varpath(name,timeline):
        return os.path.join(timeline, name)

    def traj_path(dir,map, prefix):
        return os.path.join(dir,prefix+'-rlx',map,'BMU.var')

    def frz_path(dir,map,timestep):
        return os.path.join(dir,'zfrz-%08d-out'%timestep, map, 'BMU.var')

    #open maps

    weight_dict = dict()
    r,c = find_best_rowcol(len(maps))
    fig,ax = plt.subplots(c,r,squeeze = False)
    fig2,ax2 = plt.subplots(len(maps),2,squeeze = False)
    ax=np.reshape(ax,(c,r))
    print(maps)
    for i,m in enumerate(maps):
        col,row = int(i/c), i%c
        #weights
        weights = os.listdir(os.path.join(path,m))
        wf= cx.variable.Realize(varpath('We-0.var',os.path.join(path,m)))
        wf.open()
        im = ax[row,col].imshow(wf[timestep],extent=[0,1,1,0])
        fig.colorbar(im,ax=ax[row,col])
        wf.close()


        wfc= cx.variable.Realize(varpath('Wc-0.var',os.path.join(path,m)))
        wfc.open()
        print(wfc[timestep].shape)
        imc= ax2[i,0].imshow(wfc[timestep][:,:,0],extent=[0,1,1,0])
        imc2= ax2[i,1].imshow(wfc[timestep][:,:,1],extent=[0,1,1,0])
        fig.colorbar(imc,ax=ax2[i,0])
        fig.colorbar(imc2,ax=ax2[i,1])
        wfc.close()





        for k in range(1,ntraj+1):
            with cx.variable.Realize(traj_path(dir,m,prefix+"-%03d"%k)) as b1:
                traj1 = np.array([b1[i] for i in range(b1.time_range()[1]+1)])
                print(traj1.shape)
                ax[row,col].plot(traj1[:,0], traj1[:,1],'-*',alpha=0.6)
                ax[row,col].scatter(traj1[-1,0], traj1[-1,1],c='k',s=50)
                ax2[i,0].plot(traj1[:,0], traj1[:,1],'-*')
                ax2[i,0].scatter(traj1[-1,0], traj1[-1,1],c='k',s=50)
        ax[row,col].set_title(f'Map {m}')

        with(cx.variable.Realize(frz_path(dir,m,timestep))) as bmu:
            b = bmu[timeinp]
            ax[row,col].scatter(b[0],b[1],s=150,c='r')

        #relaxation traj

    fig.suptitle(f'experience {dir }')


    #plot the inputs according to BMU

    plt.show()
