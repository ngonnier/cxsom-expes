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
        sys.exit("usage: python3 plot_weights.py <data_dir> <timestep> <prefix> <inputs> ")
    dir= sys.argv[1]
    timestep = int(sys.argv[2])
    prefix = sys.argv[3]
    inputs = sys.argv[4:]
    path = os.path.join(dir, 'wgt')
    maps = os.listdir(path)
    inp_path = os.path.join(dir,'ztest-in')
        #inputs = os.listdir(inp_path)


    def varpath(name,timeline):
        return os.path.join(timeline, name)

    #open inputs
    input_dict=dict()
    for inp in inputs:
        with cx.variable.Realize(os.path.join(inp_path,inp+".var")) as input:
            r = input.time_range()
            input_dict[inp] = np.array([input[at] for at in range(r[0],r[1])])
    #open maps

    weight_dict = dict()
    r,c = find_best_rowcol(len(maps))
    fig,ax = plt.subplots(r,c,squeeze = False)
    fig2,ax2 = plt.subplots(len(maps),2)
    ax=np.reshape(ax,(r,c))
    print(maps)
    for i,m in enumerate(maps):
        col_inp = iter(['green','magenta','black','grey'])
        row,col = int(i/c), i%c

        try:
            with cx.variable.Realize(os.path.join(directories,prefix,m,'BMU.var')) as bmu:
                r = bmu.time_range()
                bmus = np.array([bmu[at] for at in range(r[0],r[1])])
        except:
            pass

        weights = os.listdir(os.path.join(path,m))
        wf= cx.variable.Realize(varpath('We-0.var',os.path.join(path,m)))
        wf.open()
        im = ax[row,col].imshow(wf[timestep])
        fig.colorbar(im,ax=ax[row,col])
        wf.close()

        wfc= cx.variable.Realize(varpath('Wc-0.var',os.path.join(path,m)))
        wfc.open()
        print(wfc[timestep].shape)
        imc= ax2[i,0].imshow(wfc[timestep][:,:,0])
        imc2= ax2[i,1].imshow(wfc[timestep][:,:,1])
        fig.colorbar(imc,ax=ax2[i,0])
        fig.colorbar(imc2,ax=ax2[i,1])
        wfc.close()


        ax2[i,0].set_title(f'Map {m}, context weights 0')
        ax2[i,1].set_title(f'Map {m}, context weights 1')

    fig.suptitle(f'experience {dir }')


    #plot the inputs according to BMU

    plt.show()
