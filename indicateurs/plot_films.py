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

map_size = 500

if __name__=="__main__":
    if len(sys.argv)<3:
            sys.exit("Usage: python3 plot_films <path_to_dir> <path_to_result>")
    else:
        dir = sys.argv[1]
        path_results = sys.argv[2]

    map_names = ["M1","M2","M3"]
    input_names = ["I1","I2","I3"]
    inputs_0 = np.zeros((100,3))
    path_inp_0 = os.path.join(dir,"input-test-0")
    for i in range(len(input_names)):
        with cx.variable.Bind(os.path.join(path_inp_0,input_names[i]+'.var')) as inp:
            for j in range(100):
                inputs_0[j,i] = inp[j]
    inputs_1 = np.zeros((100,3))
    path_inp_1 = os.path.join(dir,"input-test-1")
    for i in range(len(input_names)):
        with cx.variable.Bind(os.path.join(path_inp_1,input_names[i]+'.var')) as inp:
            for j in range(100):
                inputs_1[j,i] = inp[j]
    inputs_2 = np.zeros((100,3))
    path_inp_2 = os.path.join(dir,"input-test-2")
    for i in range(len(input_names)):
        with cx.variable.Bind(os.path.join(path_inp_2,input_names[i]+'.var')) as inp:
            for j in range(100):
                inputs_2[j,i] = inp[j]
    #tests_numbers_0 = list(range(1,200,10))
    #tests_numbers_1 = list(range(50,2000,100))
    #tests_numbers_2 =list(range(0,13500,500))
    #test_numbers = sorted(tests_numbers_0+tests_numbers_1+tests_numbers_2)
    test_numbers = range(0,10000,100)
    #tests de 0 à 10000
    for i,idx in enumerate(test_numbers):
        fig,axes = plt.subplots(3,2)
        for j,m in enumerate(map_names):
            bmu,wbmu,weight = get_tests("test-"+str(idx),idx,dir,m)
            axes[j,0].plot(np.linspace(0,1,500),weight['We'],'r')
            axes[j,0].plot(np.linspace(0,1,500),weight['Wc-0'],'g')
            try:
                axes[j,0].plot(np.linspace(0,1,500),weight['Wc-1'],'b')
            except:
                pass
            fig.suptitle(f'Iteration {idx:04d}')
            print(len(wbmu))
            if(idx<10000):
                axes[j,1].scatter(inputs_0[:,j],wbmu[:100])
            elif(idx<15000):
                axes[j,1].scatter(inputs_1[:,j],wbmu[:100])
            else:
                axes[j,1].scatter(inputs_2[:,j],wbmu[:100])
        fig.savefig(os.path.join('results',path_results,f'film-full001-{i:03d}.png'))
        plt.close()
