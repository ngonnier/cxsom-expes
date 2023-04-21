import pycxsom as cx
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

cache_size = 2
file_size = 250000
MAP_SIZE = 500

if __name__=="__main__":
    if(len(sys.argv)<2):
        sys.exit("Usage: python3 fill_bmu.py <path_to_dir>")
    else:
        dir = sys.argv[1]

def varpath(var):
    return os.path.join(dir,'bmus-rlx',var+'.var')

with cx.variable.Realize(varpath('P1'), cx.typing.make('Pos1D'),cache_size, file_size) as p1:
    with cx.variable.Realize(varpath('P2'), cx.typing.make('Pos1D'),cache_size, file_size) as p2:
        for i in range(MAP_SIZE):
            for j in range(MAP_SIZE):
                p1 += i/MAP_SIZE
                p2 += j/MAP_SIZE
