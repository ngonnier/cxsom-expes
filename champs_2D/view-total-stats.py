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
        sys.exit("usage: python3 plot_weights.py <data_dirs>")
    paths = sys.argv[1:]
    fig,axes = plt.subplots(2,1)
    for path in paths:
        dirs= os.listdir(path)
        tot_cvg = []
        tot_mean=[]
        for dir in dirs:
            print(dir)
            exps = os.listdir(os.path.join(path,dir))
            frz_dirs = list(filter(lambda e:re.match(r'zfrz-\d{8}-rlx', e), exps))
            listcvg = []
            listmean = []
            for xp in frz_dirs:
                with cx.variable.Realize(os.path.join(path,dir,xp,'Cvg.var')) as cvg:
                    conv = [cvg[i] for i in range(cvg.time_range()[1])]
                    num_it = re.search(r'\d{8}',xp)
                    num_it = int(num_it.group(0))
                    listcvg.append([count_relax(conv),num_it])
                    listmean.append([mean_relax(conv),num_it])
            listcvg.sort(key=lambda e:e[1])
            listmean.sort(key=lambda e:e[1])
            tot_cvg.append([e[0] for e in listcvg])
            tot_mean.append([e[0] for e in listmean])

        tot_cvg = np.array(tot_cvg)
        tot_mean = np.array(tot_mean)
        mean_cvg = np.mean(tot_cvg,axis=0)
        mean_mean= np.mean(tot_mean,axis=0)
        std_cvg = np.std(tot_cvg,axis=0)
        std_mean = np.std(tot_mean,axis=0)
        nexp = len(tot_cvg[0])
        #mean_cvg = [sum([exp[i] for exp in tot_cvg])/len(tot_cvg) for i in range(0,nexp)]
        #mean_mean = [sum([exp[i] for exp in tot_mean])/len(tot_mean) for i in range(0,nexp)]




        axes[0].plot([e[1] for e in listcvg],mean_cvg,'-',label=f'{path}')
        axes[0].fill_between([e[1] for e in listcvg],mean_cvg+std_cvg,mean_cvg-std_cvg,'-',alpha=0.5,label=f'{path}')
        axes[1].plot([e[1] for e in listcvg],mean_mean,'-',label= f'{path}')
        axes[1].fill_between([e[1] for e in listcvg],mean_mean+std_mean,mean_mean-std_mean,'-',alpha=0.5,label=f'{path}')
        axes[0].set_xlabel('timestep')
        axes[1].set_xlabel('timestep')
        axes[0].set_ylabel('convergence rate %')
        axes[1].set_ylabel('mean iterations number')
        axes[0].legend()
        axes[1].legend()
    plt.show()
