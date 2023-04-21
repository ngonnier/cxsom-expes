import pycxsom as cx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.image as mpimg
import os
import sys
from correlation_ratio import *
MAP_SIZE = 100
from PIL import Image
import colorsys
import math
from colormap import *
from plot_weights_2D import plot_gridview,find_best_rowcol,colormap_a,colormap_l
if __name__ == "__main__":
    if len(sys.argv)<3:
        sys.exit("usage: python3 plot_weights.py <test_number>  <data dir> <inputs> ")
    tnum = int(sys.argv[1])
    directories = sys.argv[2]
    inputs = sys.argv[3:]

    path = os.path.join(directories, 'out')
    maps = os.listdir(path)
    inp_path = os.path.join(directories,'ztest-in')
    analysis_prefix = "zfrz"

    def varpath(name,timeline):
        return os.path.join(timeline, name)

    #open inputs
    input_dict=dict()
    for inp in inputs:
        with cx.variable.Realize(os.path.join(inp_path,inp+".var")) as input:
            r = input.time_range()
            input_dict[inp] = np.array([input[at] for at in range(r[0],r[1])])
    #open maps

    def plot_image(timestep):
        c,r = find_best_rowcol(len(maps))
        fig_grid,ax_grid = plt.subplots(r,c,figsize=(20,10),layout='tight')
        ax_grid=np.reshape(ax_grid,(r,c))


        for i,m in enumerate(maps):
            col_inp = iter(['green','magenta','black','grey'])
            row,col = int(i/c), i%c

            if m == 'M1':
                key = 'I1'
                color = 'k'
            elif m == 'M2':
                key = 'I2'
                color = 'w'
            else:
                key = 'I3'

            weight_dict = dict()
            weights = os.listdir(os.path.join(directories,'wgt',m))
            for w in weights:
                wf= cx.variable.Realize(varpath(w,os.path.join(directories,analysis_prefix+f'-%04d-wgt'%timestep,m)))
                wf.open()
                weight_dict[w] = wf[0]
                wf.close()


            ax_grid[row,col].scatter(input_dict['I1'][:,0],input_dict['I1'][:,1])
            plot_gridview(weight_dict['We-0.var'],ax_grid[row,col], 'k',step=1)
            ax_grid[row,col].set_title(f'$\omega_e$, carte $M^{i+1}$')

        return fig_grid

    fig_grid =  plot_image(tnum)
    fig_grid.savefig(f'{directories}/weights_externe-%06d.png'%tnum)
    plt.show()
