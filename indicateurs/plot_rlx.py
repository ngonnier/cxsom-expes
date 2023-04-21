import sys
import matplotlib.pyplot as plt
from plot_func import *

def get_rlx(dir,map_name):
    path = os.path.join(dir,'zfrz-rlx','Cvg.var')
    with cx.variable.Realize(path) as frlx:
        r = frlx.time_range()
        rlx = np.zeros(r[1])
        for i in range(r[1]):
            rlx[i] = frlx[i]
    return rlx


def get_rlx_test(dir,test_name,map_name):
    path = os.path.join(dir,'rlx-test-'+str(test_name),map_name,'Cvg.var')
    with cx.variable.Realize(path) as frlx:
        r = frlx.time_range()
        rlx = np.zeros(r[1])
        for i in range(r[1]):
            rlx[i] = frlx[i]
    return rlx

if __name__ == "__main__":

    if(len(sys.argv)<5):
        sys.exit("Usage: python3 plot_rlx.py <path_to_dir> <prefix> <test> <map name>")
        break

    dir = sys.argv[1]
    prefix = sys.argv[2]
    num = sys.argv[3]
    map_name = sys.argv[4]

    #plot learning rlx
    plt.figure()
    rlx = get_rlx(dir,map_name)

    plt.hist(list(rlx),200)
    plt.figure()
    plt.plot(rlx)

    #plot test rlx
    test_name = prefix+"-"+num
    #rlx1 = get_rlx_test(dir,test_name,map_name)
    plt.figure()
    #plt.hist(rlx1,100)
    plt.figure()
    #plt.plot(rlx1)
    plt.show()
