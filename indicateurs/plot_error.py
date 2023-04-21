import os
import time
import numpy as np
import pycxsom as cx
import matplotlib as mpl
import matplotlib.pyplot as plt

dir = './data'

def get_tests_inputs(map_name):
        path = os.path.join(dir,'input',map_name+'.var')
        u_path = os.path.join(dir,'input','U.var')
        with cx.variable.Bind(path) as X:
            with cx.variable.Bind(u_path) as U:
                x = np.zeros((X.duration(),X.datatype.shape()),dtype=np.float64)
                u = np.zeros(X.duration())
                for i in range(X.duration()):
                    x[i,:] = X[i]
                    u[i] = U[i]
        return x,u
def get_bmus(map_name):
    path = os.path.join(dir,'rlx',map_name,'BMU.var')
    with cx.variable.Bind(path) as X:
        x = np.zeros(X.duration())
        for i in range(X.duration()):
            x[i] = X[i]
    return x

def get_final_weights(map_name):
        w_path = os.path.join(dir,'wgt',map_name)
        print(os.listdir(w_path))
        files = [f for f in os.listdir(w_path)]
        final_weights = dict()
        for f in files:
            var_path = os.path.join(w_path,f)
            with cx.variable.Bind(var_path) as w:
                weights = w[w.duration()-1]
                var_name = f[:-4]
                final_weights[var_name] = weights
        return final_weights

def get_bmus_weights(we,bmu):
    return np.array([we[int(b*500)] for b in bmu])

if __name__ == '__main__':

    x,u = get_tests_inputs('X')
    y,_ = get_tests_inputs('Y')
    z,_ = get_tests_inputs('Z')

    bmux = get_bmus('X')
    bmuy = get_bmus('Y')
    bmuz = get_bmus('Z')

    wx = get_final_weights('X')
    wy = get_final_weights('Y')
    wz = get_final_weights('Z')

    wbmux = get_bmus_weights(wx['We'],bmux)
    wbmuy = get_bmus_weights(wy['We'],bmuy)
    wbmuz = get_bmus_weights(wz['We'],bmuz)


    fig,axes = plt.subplots(3,2)
    axes[0,0].scatter(x[:,0],wbmux[:,0])
    axes[0,1].scatter(x[:,1],wbmux[:,1])
    axes[1,0].scatter(y[:,0],wbmuy[:,0])
    axes[1,1].scatter(y[:,1],wbmuy[:,1])
    axes[2,0].scatter(y[:,2],wbmuy[:,2])
    axes[2,1].scatter(z,wbmuz)

    plt.show()
