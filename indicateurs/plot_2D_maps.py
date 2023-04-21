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

    #plot map weights

    print(m1.final_weights['We'].shape)
    plt.figure()
    plt.title('M1')

    plt.show()
