import os
import time
import numpy as np
import pycxsom as cx
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import sys

cache = 2
filesize = 20000
nmaps =  2
class Data:
    def __init__(self,dir,map_name,input_name, prefix = ''):
        self.dir = dir
        self.map_name = map_name
        self.input_name = input_name
        self.prefix = prefix
        self.final_weights = self._get_final_weights()
        #self.inputs = self._get_inputs()
        #self.bmus = self._get_bmus()
        self.test_in, self.test_u,self.test_v = self._get_tests_inputs()
        self.test_bmus = self._get_test_bmus(0)
        self.test_bmus_closed = self._get_test_bmus_closed(nmaps)
        self.test_bmus_closed_2 = self._get_test_bmus_closed_2([10])
        self.rlx = self._get_rlx()

    def _get_final_weights(self):
        w_path = os.path.join(self.dir,'wgt',self.map_name)
        files = [f for f in os.listdir(w_path)]
        final_weights = dict()
        for f in files:
            var_path = os.path.join(w_path,f)
            with cx.variable.Realize(var_path) as w:
                r = w.time_range()
                weights = w[r[1]-1]
                var_name = f[:-4]
                final_weights[var_name] = weights
        return final_weights

    def _get_bmus(self):
        path_bmu = os.path.join(self.dir,'rlx',self.map_name,'BMU.var')
        with cx.variable.Realize(path_bmu) as fbmu:
            r = fbmu.time_range()
            bmus = np.zeros(r[1],dtype=object)
            for i in range(r[0],r[1]):
                bmus[i] = fbmu[i]
        return bmus


    def _get_test_bmus(self,test_num):
        path_bmu = os.path.join(self.dir,'rlx-test-'+self.prefix+str(test_num),self.map_name,'BMU.var')
        with cx.variable.Realize(path_bmu) as fbmu:
            r = fbmu.time_range()
            bmus = np.zeros(r[1],dtype=object)
            for i in range(r[1]):
                bmus[i] = fbmu[i]
        return bmus

    def _get_test_bmus_closed(self,num_tests):
        tests = dict()
        for test_num in range(1,num_tests+1):
            path_bmu = os.path.join(self.dir,'rlx-test-'+self.prefix+str(test_num),self.map_name,'BMU.var')
            try:
                with cx.variable.Realize(path_bmu) as fbmu:
                    r = fbmu.time_range()
                    bmus = np.zeros(r[1],dtype=object)
                    for i in range(r[1]):
                        bmus[i] = fbmu[i]
                        tests[test_num] = bmus
            except:
                return []
        return tests

    def _get_test_bmus_closed_2(self,num_tests):
        tests = dict()
        for test_num in num_tests:
            path_bmu = os.path.join(self.dir,'rlx-test-2-closed-'+self.prefix+str(test_num),self.map_name,'BMU.var')
            try:
                with cx.variable.Realize(path_bmu) as fbmu:
                    r = fbmu.time_range()
                    bmus = np.zeros(fbmu.time_range(),dtype=object)
                    for i in range(r[0],r[1]):
                        bmus[i] = fbmu[i]
                        tests[test_num] = bmus
            except:
                return []
        return tests


    def _get_inputs(self):
        path = os.path.join(self.dir,'input',self.input_name+'.var')
        try:
            with cx.variable.Realize(path) as X:
                r = X.time_range()
                x = np.zeros(X.time_range(),dtype=object)
                for i in range(r[0],r[1]):
                    x[i] = X[i]
            return x
        except:
            return []

    def _get_tests_inputs(self):
        path = os.path.join(self.dir,'ztest-in',self.input_name+'.var')
        u_path = os.path.join(self.dir,'ztest-in','U.var')
        v_path = os.path.join(self.dir,'ztest-in','V.var')
        with cx.variable.Realize(path) as X:
            try:
                with cx.variable.Realize(u_path) as U:
                    try:
                        with cx.variable.Realize(v_path) as V:
                            r = X.time_range()
                            x = np.zeros(r[1],dtype=object)
                            u = np.zeros(r[1])
                            v = np.zeros(r[1])
                            for i in range(r[1]):
                                x[i] = X[i]
                                u[i] = U[i]
                                v[i] = V[i]
                    except:
                        r = X.time_range()
                        x = np.zeros(r[1],dtype=object)
                        u = np.zeros(r[1])
                        v = np.zeros(r[1])
                        for i in range(r[1]):
                            x[i] = X[i]
                            u[i] = U[i]
            except:
                r = X.time_range()
                x = np.zeros(r[1],dtype=object)
                u = np.zeros(r[1])
                v = np.zeros(r[1])
                for i in range(r[1]):
                    x[i] = X[i]


        return x,u,v


    def _get_rlx(self):
        path = os.path.join(self.dir,'rlx',self.map_name,'Cvg.var')
        try:
            with cx.variable.Realize(path) as frlx:
                r = frlx.time_range()
                rlx = np.zeros(r[1])
                for i in range(r[1]):
                    rlx[i] = frlx[i]
            return rlx
        except:
            return None


    """
    def get_weights(self):
        w_path = os.path.join(self.dir,self.prefix+'wgt',self.map_name)
        files = [f for f in os.listdir(w_path) if os.path.isfile(f)]
        final_weights = dict()
        for f in files:
            var_path = os.path.join(path,f)
            with cx.variable.Realize(var_path) as w:
                map_size = len(w[w.time_range()])
                weights = np.zeros(w.time_range(),map_size)
                for i in range(w.time_range()):
                    weights[i,:] = w[i]
        return weights
        """




def plot_final_weights(data,ax):
    for key,w in data.final_weights.items():
        ax.plot(np.linspace(0,1,len(w)),w,label=key)

def plot_inputs(data,ax):
    ax.plot(range(len(data.inputs)), data.inputs)

def scatter_inputs_bmus(data,ax):
    ax.scatter(data.test_bmu,data.x)

def scatter_error(data,ax):
        we = data.final_weights['We']
        bmu = data.test_bmus
        inp = data.test_in
        map_size = len(we)

        wbmu = [we[math.floor(elt*map_size)] if(elt<1) else we[-1] for elt in bmu]
        ax.scatter(inp[:len(wbmu)],wbmu)

def scatter_error_cont(data,real_bmus,cnum,ax):
        wc = data.final_weights['Wc-'+str(cnum)]
        bmu = data.test_bmus
        map_size = len(wc)
        wbmu = [wc[math.floor(elt*map_size)] if(elt<1) else wc[-1] for elt in bmu]
        ax.scatter(real_bmus[:len(wbmu)],wbmu)


def scatter_error_closed(data,ax,num,is_closed):
        we = data.final_weights['We']
        bmu = data.test_bmus_closed[num]
        inp = data.test_in
        map_size = len(we)

        wbmu = [we[math.floor(elt*map_size)] if(elt<1) else we[-1] for elt in bmu]
        ax.scatter(inp[:len(wbmu)],wbmu)
        if(is_closed):
            ax.set_facecolor("#f7f498")

def scatter_error_closed_2(data,ax,num,is_closed):
        we = data.final_weights['We']
        bmu = data.test_bmus_closed_2[num]
        inp = data.test_in
        map_size = len(we)
        wbmu = [we[math.floor(elt*map_size)] if(elt<1) else we[-1] for elt in bmu]
        ax.scatter(inp[:len(wbmu)],wbmu)
        if(is_closed):
            ax.set_facecolor("#f7f498")

def mse(data,num,is_closed):
    we = data.final_weights['We']
    map_size = len(we)
    bmu = []
    if(is_closed):
        bmu = data.test_bmus_closed[num]
    else:
        bmu = data.test_bmus

    inp = data.test_in
    wbmu = [we[math.floor(elt*map_size)] if(elt<1) else we[-1] for elt in bmu]
    se = [(w - i)**2 for w,i in zip(wbmu,inp)]
    mse = np.sqrt(sum(se))/len(se)
    return mse


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


def get_tests(test_name,test_number,dir,map_name):
    #get BMUs
    path_bmu = os.path.join(dir,"zfrz-"+test_name,map_name,'BMU.var')
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
            weights_i = w[test_number]
            var_name = f[:-4]
            weights[var_name] = weights_i
    if('We-0' in weights):
        we = weights['We-0']
        map_size=we.shape
        print(map_size)
        #get BMU weights
        if(len(map_size)==1):
            wbmu = [we[math.floor(elt*(map_size[0]-1))] for elt in bmus]
        else:
            wbmu=[]
            for elt in bmus:
                bmu_x = int(elt[0]*map_size[0]-1)
                bmu_y = int(elt[1]*map_size[1]-1)
                wbmu.append(we[bmu_x,bmu_y])
    else:
        wbmu = []

    return bmus,wbmu,weights

def get_inputs(dir):
    path_in = os.path.join(dir,"ztest-in")
    files = os.listdir(path_in)
    inputs = dict()
    for inp in files:
        inf= cx.variable.Realize(os.path.join(path_in,inp))
        inf.open()
        r = inf.time_range()
        temp = np.zeros(r[1])
        for i in range(r[1]):
            temp[i] = inf[i]
        inf.close()
        inputs[f'{inp[:-4]}'] = temp
    return inputs

def get_weights(number,dir,map_name):
    path = os.path.join(dir, 'wgt',map_name)
    def varpath(name,timeline):
        return os.path.join(timeline, name)
    mapdict=dict()
    weights = os.listdir(path)
    for w in weights:
        wf= cx.variable.Realize(varpath(w,path))
        wf.open()
        r = wf.time_range()
        if(r[1]>1):
            mapdict[f'{w[:-4]}'] = wf[number]
        wf.close()
    return mapdict


if __name__ == '__main__':
    if len(sys.argv)<4:
        sys.exit("Usage: python3 inputs.py <path_to_dir> <prefix> <test> <map names>")
    else:
        dir = sys.argv[1]
        test = sys.argv[3]
    map_names = sys.argv[4:]
    input_names = ['I1','I2','I3','I4','I5','I6']
    #closed_tests = [(0,2),(1,5),(3,4)]
    prefix = sys.argv[2]
    plt.rc('text',usetex=True)
    c,r = find_best_rowcol(len(map_names))
    fig,axes = plt.subplots(r,c)
    axes = np.reshape(axes,(r,c))
    fig2,axes2 = plt.subplots(r,c)
    axes2 = np.reshape(axes2,(r,c))
    fig3,axes3 =plt.subplots(r,c)
    axes3 = np.reshape(axes3,(r,c))
    inp_combi = int(math.factorial(len(map_names))/(2*math.factorial(len(map_names)-2)))
    r1,c1 = find_best_rowcol(inp_combi)
    fig4,axes4 = plt.subplots(r,c)
    axes4 = np.reshape(axes4,(r,c))

    tab_mse = np.zeros((len(map_names)+1,len(map_names)))
    map_data = []
    for i,(map,input) in enumerate(zip(map_names,input_names)):
        dat = Data(dir,map,input,prefix)
        map_data.append(dat)


    for i,dat in enumerate(map_data):
        row,col = int(i/c), i%c
        plot_final_weights(dat,axes[row,col])
        axes[row,col].scatter(dat.test_bmus,dat.test_in,color='k',label='inputs')
        axes[row,col].legend()
        axes[row,col].set_title(f'Map {i}')
        scatter_error(dat,axes2[row,col])
        axes2[row,col].set_title(f"Map {i}")
        fig2.suptitle("test with all inputs")
        tab_mse[0,i]=mse(dat,0,False)
        axes3[row,col].scatter(dat.test_bmus,dat.test_u)
        axes3[row,col].set_title(f"Map {i}")
        fig3.suptitle("U according to BMU positions")
        plot_final_weights(dat,axes4[row,col])
        #fig4.suptitle("V according to BMU positions")
        #print(dat.test_bmus_closed[1])
        #axes4[row,col].scatter(dat.test_bmus_closed[1],dat.test_in,color='k',label = 'inputs')
        #axes4[row,col].set_title(f"Map {i}")
        axes4[row,col].legend()
        fig4.suptitle("Map X closed")

    for j in range(1,len(map_data)+1):
        fig,axes=plt.subplots(r,c)
        axes = np.reshape(axes,(r,c))
        for i,dat in enumerate(map_data):
            row,col = int(i/c),i%c
            scatter_error_closed(dat,axes[row,col],j,i==j-1)
            #tab_mse[j,i]=mse(dat,j-1,True)
            axes[row,col].set_title(f'Map {i}')
        fig.suptitle(f"Test {j}: Map {j-1} closed")


    """
    for j in range(0,len(closed_tests)):
        fig,axes=plt.subplots(r,c)
        for i,dat in enumerate(map_data):
            row,col = int(i/c),i%c
            scatter_error_closed_2(dat,axes[row,col],j,i==closed_tests[j][0] or i==closed_tests[j][1])
            #tab_mse[j,i]=mse(dat,j-1,True)
            axes[row,col].set_title(f'Map {i}')
        fig.suptitle(f"Test {j}: Maps {closed_tests[j][0]} and {closed_tests[j][1]} closed")
        """




    plt.show()
