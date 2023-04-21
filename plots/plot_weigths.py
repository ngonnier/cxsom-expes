import pycxsom as cx
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from correlation_ratio import *
import re

plt.rc('text', usetex=True)
plt.rcParams['font.size'] = '16'



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

def plot_fft(weights_dict,w_name,ax):
    sp = np.fft.fft(weight_dict[w_name])
    freq = np.fft.fftfreq(weight_dict[w_name].shape[-1])
    ax.plot(freq,np.abs(sp))

def plot_activite(path_map,ax,time_inp):
    #cti = os.listdir(path_map)
    acti = ['Ac.var']
    acti_reel = os.listdir(path_map)
    for a in acti:
        if a in acti_reel:
            with cx.variable.Realize(os.path.join(path_map,a)) as a_file:
                print(a_file.time_range())
                a_tab = a_file[time_inp]
                ax.fill_between(np.linspace(0,1,len(a_tab)), a_tab, alpha = 0.4, color='m',  label = a[:-4])
        ax.legend()

def to_name(varname,map, ncartes=2):
    w_name = re.match('W',varname)
    if w_name:
        map_name = int(re.search('\d+',map)[0])
        if varname == 'We-0.var':
            return '$\\omega_e^{(%d)}$' % map_name
        elif varname == 'Wc-0.var':
            if ncartes==2:
                return '$\\omega_c^{(%d)}$' % map_name
            else:
                return '$\\omega_{c_0}^{(%d)}$' % map_name
        else:
            print(varname)
            numw = re.search('\d+',varname)[0]
            return '$\\omega_{c_{%s}}^{(%d)}$' % (numw,map_name)

    else:
        input_name = re.match('I\d',varname)
        if input_name:
            print(varname)
            num = re.search('\d+',varname)
            return '$X^{(%s)}$' % num[0]
        else:
            return f'${varname}$'

def name_to_input_name(name):
        if name == 'cmdy':
            return "$\\rho$"
        elif name == 'ct':
            return '$\\varphi$'
        elif name =='odomy':
            return '$v$'
        elif name == 'vpx':
            return '$x$'
        else:
            input_name = re.match('I\d',varname)
            if input_name:
                num = re.search('\d+',varname)
                return '$X^{(%s)}$' % num[0]
            else:
                return f'${varname}$'

def name_to_map(name):
    if name == 'cmdy':
        return "M1"
    elif name == 'ct':
        return 'M2'
    elif name =='odomy':
        return 'M3'
    elif name == 'vpx':
        return 'M4'
    else:

        num = re.search('\d+',name)[0]
        return f'M{num}'



if __name__ == "__main__":

    if len(sys.argv)<3:
        sys.exit("usage: python3 plot_weights.py <test_number> <closed>  <opt> <time_inp> <data dir> <inputs> ")
    tnum = int(sys.argv[1])
    closed = int(sys.argv[2])
    opt = sys.argv[3]
    time_inp = int(sys.argv[4])
    directories = sys.argv[5]
    inputs = sys.argv[6:]
    path = os.path.join(directories, 'wgt')

    inp_path = os.path.join(directories,'ztest-in')
    #inputs = [ elt[:-4] for elt in os.listdir(inp_path)]
    maps = [name_to_map(inp) for inp in inputs]

    analysis_prefix = "zfrz"
    if closed >= 1:
        print(closed)
        analysis_prefix = f"zclosed-{closed}"

    max_bmu=999

    def varpath(name,timeline):
        return os.path.join(timeline, name)

    #open inputs
    input_dict=dict()
    for inp in inputs:
        with cx.variable.Realize(os.path.join(inp_path,inp+".var")) as input:
            r = input.time_range()
            input_dict[inp] = np.array([input[at] for at in range(r[0],max_bmu)])

    #open U
    #if('U.var') in inputs:
    with cx.variable.Realize(os.path.join(inp_path,"U.var")) as input:
        r = input.time_range()
        input_dict['U'] = np.array([input[at] for at in range(r[0],max_bmu)])

    if opt=='image':
        weight_dict = dict()
        c,r = find_best_rowcol(len(maps))
        fig,ax = plt.subplots(r,c,squeeze = False,figsize=(5*c,2.5*r+2))
        fig2,ax2 = plt.subplots(r,c,squeeze = False)
        ax=np.reshape(ax,(r,c))
        ax2 = np.reshape(ax2,(r,c))
        print(maps)
        for i,m in enumerate(maps):
            col_inp = iter(['tab:red','tab:green','tab:brown','grey'])
            row,col = int(i/c), i%c

            try:
                print(os.path.join(directories,analysis_prefix+f'-%04d-out'%tnum,m,'BMU.var'))
                with cx.variable.Realize(os.path.join(directories,analysis_prefix+f'-%04d-out'%tnum,m,'BMU.var')) as bmu:
                    r = bmu.time_range()
                    bmus = np.array([bmu[at] for at in range(r[0],r[1])])
            except:
                pass

            acti_path = os.path.join(directories,analysis_prefix+f'-%04d-rlx'%tnum,m)
            weights = os.listdir(os.path.join(path,m))

            #DRONE-specific
            # if(m == name_to_map('cmdy')):
            #     plot_activite(acti_path,ax[row,col],time_inp)
            #     ax[row,col].vlines(x = bmus[time_inp], linewidth=2, color= 'k', linestyle='--' ,ymin=0, ymax=1.1,label = '$\Pi$')


            for w in weights:
                wf= cx.variable.Realize(varpath(w,os.path.join(path,m)))
                wf.open()
                p = np.linspace(0,1,len(wf[tnum]))
                ax[row,col].plot(p, wf[tnum],'-', markerfacecolor = 'None', label=to_name(w,m),zorder=1)
                #ax[row,col].scatter(np.linspace(0,1,len(wf[tnum])),wf[tnum],facecolors='none',edgecolors='red',alpha=0.5)
                weight_dict[w] = wf[tnum]
                wf.close()

            for key in input_dict.keys():
                if(key!='U'):
                    ax[row,col].set_xlabel('Positions')
                    ax[row,col].set_ylabel('Poids')
                    print(len(bmus))
                    print(len(input_dict[key]))
                    ax[row,col].scatter(bmus[:max_bmu],input_dict[key][:max_bmu],alpha=0.4,s = 10, label=to_name(key,i,ncartes=len(maps)),color=next(col_inp))
                    #ax[row,col].scatter(bmus[11],input_dict[key][11],alpha=0.4,s = 20,c='k')
                    print(f'{key}: BMU {bmus[time_inp]}, Input {input_dict[key][time_inp]}')
                    #cr_in, phi_in, phi_edges_in = correlation_ratio(input_dict['I1'],input_dict['I2'],500)
                    #print(f"eta(I1,I2 = { cr_in}")
            ax[row,col].set_xlabel('Positions $p$')
            #ax[row,col].set_ylabel('Entrées')

            #uv = np.concatenate([np.reshape(input_dict['U'], (len(input_dict['U']),1)), np.reshape(input_dict['V'], (len(input_dict['V']),1))], axis = -1)
            uv = input_dict['U']
            corr_ratio_B1U, phiB1U, phi_edges = correlation_ratio(uv,bmus,500)
            #cr, phi = mutual_information(bmus, input_dict['U'],50,250)
            ax2[row,col].scatter(bmus,input_dict['U'],alpha=0.5,label=f"{key[:-4]}",s=20)
            ax2[row,col].plot(phi_edges,phiB1U,'r',alpha=0.5)
            ax2[row,col].set_xlabel('$\Pi^{(%s)}$'% m[-1])
            ax2[row,col].set_ylabel('$U$')
            #ax2[row,col].text(0,0, f'$\\eta(U|\Pi^{i+1}) = {corr_ratio_B1U:.3f}$')
            #ax2[row,col].text(0.1,-0.3, f'$MI(U|\Pi^{i+1}) = {mi/eu:.3f}$')

            ax[row,col].legend(loc = 'center', bbox_to_anchor =(0.5,-0.4), ncol=c)
            ax[row,col].set_title('Carte $M^{(%s)}$' % m[-1])
            ax2[row,col].set_title('Carte $M^{(%s)}$' % m[-1])

        fig.tight_layout()
        fig.savefig(f'{directories}weights.svg')
        plt.show()


    if opt =='film':
        for tnum in range(0,9900,100):
            weight_dict = dict()
            fig,ax = plt.subplots(1,len(maps),squeeze = False)
            for i,m in enumerate(maps):
                with cx.variable.Realize(os.path.join(p,f'rlx-test-{tnum}',m,'BMU.var')) as bmu:
                    r = bmu.time_range()
                    bmus = np.array([bmu[at] for at in range(r[0],r[1])])
                weights = os.listdir(os.path.join(path,m))
                for w in weights:
                    wf= cx.variable.Realize(varpath(w,os.path.join(path,m)))
                    wf.open()
                    ax[0,i].plot(np.linspace(0,1,len(wf[tnum])),wf[tnum],label=f"{w[:-4]}")
                    weight_dict[w] = wf[tnum]
                    wf.close()

                # for key in input_dict.keys():
                #     if(key!='U.var'):
                #
                #         ax[0,i].scatter(bmus,input_dict[key],alpha=0.7,label=f"{key[:-4]}")

                ax[0,i].set_title(f'Map {m}')
                #ax[0,i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
                ax[0,i].set_aspect('equal')

            fig.suptitle(f'experience {directories}')
            fig.savefig(f'rcre0-{int(tnum/100)}.png')






    #plt.show()
