import os
import time
import numpy as np
import pycxsom as cx
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import math
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


def to_name(varname,map, ncartes=2):
    w_name = re.match('W',varname)
    if w_name:
        if varname == 'We-0.var':
            return '$\\omega_e^{(%d)}$' % (map+1)
        elif varname == 'Wc-0.var':
            if ncartes==2:
                return '$\\omega_c^{(%d)}$' % (map+1)
            else:
                return '$\\omega_{c_0}^{(%d)}$' % (map+1)
        else:
            numw = re.search('\d+',varname)[0]
            return '$\\omega_{c_{%s}}^{(%d)}$' % (numw, (map+1))

    else:
        input_name = re.match('I\d',varname)
        if input_name:
            num = re.search('\d+',varname)
            return '$X^{(%s)}$' % num[0]
        else:
            return f'${varname}$'


def get_tests(analysis_prefix,test_number,dir,map_name):
    #get BMUs
    path_bmu = os.path.join(dir,analysis_prefix+'-out',map_name,'BMU.var')
    print(path_bmu)
    with cx.variable.Realize(path_bmu) as fbmu:
        r = fbmu.time_range()
        bmus = np.zeros(r[1],dtype='object')
        for i in range(r[1]):
            bmus[i] = fbmu[i]
    print(map_name)
    w_path = os.path.join(dir,'wgt',map_name)
    #get weights
    files = [f for f in os.listdir(w_path)]
    weights = dict()

    for f in files:
        var_path = os.path.join(w_path,f)
        with cx.variable.Realize(var_path) as w:
            weights_i = w[tnum]
            var_name = f[:-4]
            weights[var_name] = weights_i

    if('We-0' in weights):
        we = weights['We-0']
        #get BMU weights
        map_size= we.shape
        if(len(we.shape)==1):
            wbmu = [we[int(elt*(we.shape[0]))] for elt in bmus]
        else:
             wbmu=[]
             for elt in bmus:
                 bmu_x = int(elt[0]*(map_size[0]-1))
                 bmu_y = int(elt[1]*(map_size[1]-1))
                 wbmu.append(we[bmu_y,bmu_x])
    else:
        wbmu = []

    return bmus,wbmu,weights

def get_inputs(dir):
    path_in = os.path.join(dir,"ztest-in")
    files = os.listdir(path_in)
    inputs = dict()
    for inp in files:
        print(inp)
        inf= cx.variable.Realize(os.path.join(path_in,inp))
        inf.open()
        r = inf.time_range()
        print(r)
        temp = np.zeros(r[1],dtype=object)
        for i in range(r[1]):
            temp[i] = inf[i]
        inf.close()
        inputs[f'{inp[:-4]}'] = temp
    return inputs

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
        print(name)
        num = re.search('\d+',name)[0]
        return f'M{num}'

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
            input_name = re.match('I\d',name)
            if input_name:
                num = re.search('\d+',name)[0]
                return '$X^{(%s)}$' % num[0]
            else:
                return f'${varname}$'


if __name__=="__main__":
        if len(sys.argv)<3:
            sys.exit("usage: python3 plot_error.py <analysis_prefix> <test_number> <data dir> <dim> <inputs>")
        analysis_prefix = sys.argv[1]
        tnum = int(sys.argv[2])
        directories = sys.argv[3]
        input_names = sys.argv[5:]
        dim = int(sys.argv[4])
        print(dim)
        path = os.path.join(directories, 'wgt')
        maps = os.listdir(path)
        print(maps)
        inp_path = os.path.join(directories,'ztest-in')

        bmu1,wbmu1,weights1 = get_tests(analysis_prefix,tnum,directories,"M1")
        bmu2,wbmu2,weights2 = get_tests(analysis_prefix,tnum,directories,"M2")
        try:
            bmu3,wbmu3,weights3 = get_tests(analysis_prefix,tnum,directories,"M3")
        except:
            pass

        inputs = get_inputs(directories)
        print(input_names)
        c,r = find_best_rowcol(len(input_names))
        fig,ax = plt.subplots(r,c)
        ax = ax.reshape((r,c))

        if(dim==1):
            for i, name in enumerate(input_names):
                row,col = int(i/c), i%c
                bmu1,wbmu1,weights1 = get_tests(analysis_prefix,tnum,directories,name_to_map(name))
                print(len(wbmu1))
                print(len(inputs[name]))
                print(wbmu1[11])
                print(inputs['I1'][11])
                print(inputs['I2'][11])
                min_len = min(len(wbmu1),len(inputs[name]))
                #compter le nombre de BMUs proche de la diagonale
                bmus_ok = list(filter(lambda x: abs(x[0]-x[1])<0.1, zip(list(inputs[name][0:min_len]),list(wbmu1[0:min_len]))))
                print(f'Carte {name} : {len(bmus_ok)}/{min_len} bmus proches de la diagonale à 0.1 près')
                #min_len=12
                ax[row,col].scatter(inputs[name][0:min_len],wbmu1[0:min_len],s=10)
                ax[row,col].scatter([b[0] for b in bmus_ok] , [b[1] for b in bmus_ok],s=10, c='r')
                ax[row,col].set_xlabel(name_to_input_name(name))
                ax[row,col].set_ylabel('$\\omega_e(\\Pi^{(%s)})$' % name_to_input_name(name)[1:-1])

        if(dim==2):
            min_len = min(len(wbmu1),len(inputs['I1']))
            fig,ax = plt.subplots(2,2)
            print(len(inputs['I1'][0]))
            ax[0,0].scatter([inp[0] for inp in inputs["I1"][:min_len]],[w[0] for w in wbmu1[:min_len]],s=10)
            ax[0,1].scatter([inp[1] for inp in inputs["I1"][:min_len]],[w[1] for w in wbmu1[:min_len]],s=10)
            ax[1,0].scatter([inp[0] for inp in inputs["I2"][:min_len]],[w[0] for w in wbmu2[:min_len]],s=10)
            ax[1,1].scatter([inp[1] for inp in inputs["I2"][:min_len]],[w[1] for w in wbmu2[:min_len]],s=10)

            ax[0,0].set_xlabel('$X^{(1)}|_x$')
            ax[0,1].set_xlabel('$X^{(1)}|_y$')
            ax[1,0].set_xlabel('$X^{(2)}|_x$')
            ax[1,1].set_xlabel('$X^{(2)}|_y$')
            ax[0,0].set_ylabel('$\\omega_e(\\Pi^{(1)})|_x$')
            ax[0,1].set_ylabel('$\\omega_e(\\Pi^{(1)})|_y$')
            ax[1,0].set_ylabel('$\\omega_e(\\Pi^{(2)})|_x$')
            ax[1,1].set_ylabel('$\\omega_e(\\Pi^{(2)})|_y$')

        if(dim==3):
            fig,ax = plt.subplots(3,2)
            bmu1,wbmu1,weights1 = get_tests(analysis_prefix,tnum,directories,"M1")
            bmu2,wbmu2,weights2 = get_tests(analysis_prefix,tnum,directories,"M2")
            bmu3,wbmu3,weights3 = get_tests(analysis_prefix,tnum,directories,"M3")
            ax[0,0].scatter([inp[0] for inp in inputs["I1"]],[w[0] for w in wbmu1])
            ax[0,1].scatter([inp[1] for inp in inputs["I1"]],[w[1] for w in wbmu1])
            ax[1,0].scatter([inp[0] for inp in inputs["I2"]],[w[0] for w in wbmu2])
            ax[1,1].scatter([inp[1] for inp in inputs["I2"]],[w[1] for w in wbmu2])
            ax[2,0].scatter([inp[0] for inp in inputs["I3"]],[w[0] for w in wbmu3])
            ax[2,1].scatter([inp[1] for inp in inputs["I3"]],[w[1] for w in wbmu3])
            ax[0,0].set_xlabel('$X^{(1)}|_x$')
            ax[0,1].set_xlabel('$X^{(1)}|_y$')
            ax[1,0].set_xlabel('$X^{(2)}|_x$')
            ax[1,1].set_xlabel('$X^{(2)}|_y$')
            ax[0,0].set_ylabel('$\\omega_e(\\Pi^{(1)})|_x$')
            ax[0,1].set_ylabel('$\\omega_e(\\Pi^{(1)})|_y$')
            ax[1,0].set_ylabel('$\\omega_e(\\Pi^{(2)})|_x$')
            ax[1,1].set_ylabel('$\\omega_e(\\Pi^{(2)})|_y$')
            ax[2,0].set_xlabel('$X^{(3)}|_x$')
            ax[2,1].set_xlabel('$X^{(3)}|_y$')
            ax[2,0].set_ylabel('$\\omega_e(\\Pi^{(3)})|_x$')
            ax[2,1].set_ylabel('$\\omega_e(\\Pi^{(3)})|_y$')

        fig.savefig(f'{directories}/{analysis_prefix}_error.svg')
        plt.show()
