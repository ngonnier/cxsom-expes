from plot_func import Data
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
    m3 = Data(dir,'M3','I3',prefix)
    #Plot Pi1,Pi2,Pi3

    fig = plt.figure()
    plt.rc('text',usetex = True)
    ax = fig.add_subplot(111,projection='3d')

    ax.scatter(m1.test_bmus,m2.test_bmus,m3.test_bmus)
    ax.set_xlabel('$\\Pi_1$')
    ax.set_ylabel('$\\Pi_2$')
    ax.set_zlabel('$\\Pi_3$')

    plt.show()
