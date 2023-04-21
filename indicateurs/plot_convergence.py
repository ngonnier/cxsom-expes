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
import scipy.signal as sp


import pycxsom as cx
import numpy as np
import matplotlib.pyplot as plt
import os
import sys



if __name__ == "__main__":
    if len(sys.argv)<3:
        sys.exit("usage: python3 plot_weights.py <test_number> <data dirs>")
    tnum = int(sys.argv[1])
    directories = sys.argv[2:]
    for p in directories:
        path = os.path.join(p, 'wgt')
        maps = os.listdir(path)

        def varpath(name,timeline):
            return os.path.join(timeline, name)


        fig,ax = plt.subplots(len(maps),1,squeeze = False)

        for i,m in enumerate(maps):
            weights = os.listdir(os.path.join(path,m))
            for w in weights:
                wf= cx.variable.Realize(varpath(w,os.path.join(path,m)))
                wf.open()
                #if(wf.time_range()[1]>1):
                ax[i,0].plot(range(0,len(wf[tnum])),wf[tnum],label=f'{w[:-4]}')
                wf.close()

            ax[i,0].set_title(f'Map {m}')
            ax[i,0].legend()
            fig.suptitle(f'experience {p}')

    plt.show()
