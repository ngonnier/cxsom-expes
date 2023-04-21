import os
import time
import numpy as np
import pycxsom as cx
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import sys
from plot_func import *
from inputs_sphere import value_to_uv

if __name__ == '__main__':
    if len(sys.argv)<2:
        sys.exit("Usage: python3 inputs.py <path_to_dir>")
    else:
        dir = sys.argv[1]
        
    map_names = ["M1","M2","M3","M4","M5","M6"]
    input_names = ["I1","I2","I3","I4","I5","I6"]
    closed_tests = [(0,2),(1,5),(3,4)]
    r,c = find_best_rowcol(len(map_names))
    tab_mse = np.zeros((len(map_names)+1,len(map_names)))
    map_data = []
    for i,(map,input) in enumerate(zip(map_names,input_names)):
        dat = Data(dir,map,input)
        map_data.append(dat)
    fig1,ax1 = plt.subplots()
    bmus_weights = dict()
    for map in map_data:
        we = map.final_weights['We']
        map_size = len(we)
        bmu = map.test_bmus
        wbmu = [we[math.floor(elt*map_size)] if(elt<1) else we[-1] for elt in bmu]
        bmus_weights[map.map_name] = wbmu

    nb_inp = len(map_data[0].test_u)
    uarr = np.zeros((100,2))

    for p in np.arange(0.,1.,0.01):
        idx = next((i for i,x in enumerate(map_data[0].test_bmus) if (x>=p-0.01 and x<=p+0.01)),-1)
        u1,v1= value_to_uv(bmus_weights['M1'][idx],bmus_weights['M2'][idx],bmus_weights['M3'][idx],bmus_weights['M4'][idx],bmus_weights['M5'][idx],bmus_weights['M6'][idx])
        uarr[math.floor(p*100),:] = [u1,v1]

    ax1.plot(uarr[:,0],uarr[:,1],'go-')
    plt.show()
