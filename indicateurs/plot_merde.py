from plot_func import Data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import numpy as np
import pycxsom as cx
import math

def scatter_error(data,ax,slice):
        we = data.final_weights['We']
        bmu = data.test_bmus[slice[0]:slice[1]]
        inp = data.test_in[slice[0]:slice[1]]
        map_size = len(we)
        print(map_size)
        print(len(bmu))
        print(len(inp))
        print(bmu)
        wbmu = [we[math.floor(elt*map_size)] if(elt<1) else we[-1] for elt in bmu]
        print(len(wbmu))
        ax.scatter(inp[:len(wbmu)],wbmu)

def get_realxyz(dir):
    with cx.variable.Bind(os.path.join(dir,'input-test/real.var'), cx.typing.make('Scalar')) as r:
        real = np.array([r[at] for at in range(r.duration())])
    return real

if __name__ == "__main__":
    map_names = ["M1","M2","M3"]
    input_names = ["I1","I2","I3"]
    prefix = "13999-"
    map_data = []
    dir = '3som/3som_test_merde'
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 4)
    real = np.array(get_realxyz(dir))
    for i,(map,input) in enumerate(zip(map_names,input_names)):
        dat = Data(dir,map,input,prefix)
        map_data.append(dat)
        ax1 =fig.add_subplot(gs[0,i])
        ax1.set_title('Map 1')
        ax2 =fig.add_subplot(gs[1,i])
        ax2.set_title('Map 2')
        ax3 =fig.add_subplot(gs[2,i])
        ax3.set_title('Map 3')
        axes = [ax1,ax2,ax3]
        for j in range(len(map_names)):
            if(j==i):
                we = dat.final_weights['We']
                bmu = dat.test_bmus[j*200:(j+1)*200]
                map_size = len(we)
                wbmu = [we[math.floor(elt*map_size)] if(elt<1) else we[-1] for elt in bmu]
                axes[j].scatter(real[j*200:(j+1)*200],wbmu)
                axes[j].set_facecolor( "#fcf48f")
            else:
                scatter_error(dat,axes[j],(j*200,(j+1)*200))

    ax4 = fig.add_subplot(gs[0,3])
    ax4.annotate('X random', (0.1, 0.5), xycoords='axes fraction', va='center')
    ax4.axis('off')
    ax5 = fig.add_subplot(gs[1,3])
    ax5.annotate('Y random', (0.1, 0.5), xycoords='axes fraction', va='center')
    ax5.axis('off')
    ax6 = fig.add_subplot(gs[2,3])
    ax6.annotate('Z random', (0.1, 0.5), xycoords='axes fraction', va='center')
    ax6.axis('off')


    plt.show()
