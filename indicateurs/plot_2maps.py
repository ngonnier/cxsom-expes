from plot_func import *
import sys
import matplotlib.pyplot as plt

if __name__=="__main__":
    if len(sys.argv)<2:
        sys.exit("Usage: python3 inputs.py <path_to_dir> <prefix>")
    else:
        prefix = ''
        dir = sys.argv[1]
        if len(sys.argv)>=3:
            prefix = sys.argv[2]

    m1 = Data(dir,'M1','I1',prefix)
    m2 = Data(dir,'M2','I2',prefix)
    #Plot Pi1,Pi2

    fig = plt.figure()
    plt.rc('text',usetex = True)
    ax = fig.add_subplot(111)
    ax.scatter(m1.test_bmus[1:-2],m2.test_bmus[1:-2])
    ax.set_xlabel('$\\Pi_1$')
    ax.set_ylabel('$\\Pi_2$')

    fig = plt.figure()

    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(m1.test_bmus,m2.test_bmus,m1.test_u)
    ax.set_xlabel('$\\Pi_1$')
    ax.set_ylabel('$\\Pi_2$')
    ax.set_zlabel('$U$')

    fig,axes = plt.subplots(2,1)
    plot_final_weights(m1,axes[0])
    plot_final_weights(m2,axes[1])
    fig,axes = plt.subplots(2,2)
    scatter_error_closed(m1,axes[0,0],1,1)
    scatter_error_closed(m2,axes[0,1],1,0)
    scatter_error_closed(m1,axes[1,0],2,0)
    scatter_error_closed(m2,axes[1,1],2,1)

    fig,axes=plt.subplots(2,1)
    scatter_error_cont(m1,m2.test_bmus,0,axes[0])
    axes[0].set_xlabel('$\\Pi_2$')
    axes[0].set_ylabel('M1 context weights')
    scatter_error_cont(m2,m1.test_bmus,0,axes[1])
    axes[1].set_xlabel('$\\Pi_1$')
    axes[1].set_ylabel('M2 context weights')
    plt.show()
