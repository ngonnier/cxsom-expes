import pycxsom as cx
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
def count_relax(conv):
    return len(list(filter(lambda x:x<999,conv)))/len(conv) * 100

def mean_relax(conv):
    return sum(conv) / len(conv)

if __name__ == "__main__":
    if len(sys.argv)<2:
        sys.exit("usage: python3 plot_weights.py <data_dir> <timestep>")


    dirs= sys.argv[1:]
    """
    timestep = int(sys.argv[2])

    with cx.variable.Realize(os.path.join(dir,'zfrz-%04d-rlx'%timestep,'Cvg.var')) as cvg:
        conv = [cvg[i] for i in range(cvg.time_range()[1])]

    print('taux de convergence: '+str(count_relax(conv)))
    print("nombre moyen d'itérations: "+str(mean_relax(conv)))
    """
    fig,axes = plt.subplots(2,1)
    for dir in dirs:
        exps = os.listdir(dir)
        frz_dirs = list(filter(lambda e:re.match(r'zfrz-\d{4}-rlx', e), exps))
        print(frz_dirs)
        listcvg = []
        listmean = []
        for xp in frz_dirs:
            with cx.variable.Realize(os.path.join(dir,xp,'Cvg.var')) as cvg:
                conv = [cvg[i] for i in range(cvg.time_range()[1])]
                num_it = re.search(r'\d{4}',xp)
                num_it = int(num_it.group(0))
                listcvg.append([count_relax(conv),num_it])
                listmean.append([mean_relax(conv),num_it])


        listcvg.sort(key=lambda e:e[1])
        listmean.sort(key=lambda e:e[1])

        axes[0].plot([l[1] for l in listcvg], [l[0] for l in listcvg],'o-',label=f'{dir}')
        axes[1].plot([l[1] for l in listmean], [l[0] for l in listmean],'o-',label= f'{dir}')
        axes[0].set_xlabel('timestep')
        axes[1].set_xlabel('timestep')
        axes[0].set_ylabel('convergence rate %')
        axes[1].set_ylabel('mean iterations number')

    axes[0].legend()
    axes[1].legend()
    plt.show()
