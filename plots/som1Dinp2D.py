import pycxsom as cx
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

#read weights

if __name__ == "__main__":
    if len(sys.argv)<3:
        sys.exit("usage: python3 plot_weights.py <test_number> <data dir>")
    else:
        p = sys.argv[2]
        n = int(sys.argv[1])
        path = os.path.join(p, 'wgt')

    #maps = os.listdir(path)
    def varpath(name,timeline):
        return os.path.join(timeline, name)

    varpath_inp = os.path.join(p,'input-test')
    inp_name = 'X.var'
    x= cx.variable.Realize(os.path.join(varpath_inp,inp_name), cx.typing.make('Array=2'))
    x.open()
    r = x.time_range()
    X = np.array([x[at] for at in range(r[0],r[1])])
    x.close()

    for nn in list(range(0,500,50))+list(range(1000,27000,5000)):
        w ='W.var'
        wf= cx.variable.Realize(varpath(w,path))
        wf.open()
        prot = wf[nn]
        wf.close()
        print(prot.shape)
        fig = plt.figure()

        #inputs
        plt.scatter(X[:,0],X[:,1],c = 'b',alpha=0.7)
        #points
        plt.scatter(prot[:,0], prot[:,1],c = 'k')
        #grid
        plt.plot(prot[:,0],prot[:,1], 'k')

        #save piplt.savefig('foo.png')cture
        plt.savefig(f'som-{nn}.pdf')


#plt.show()
