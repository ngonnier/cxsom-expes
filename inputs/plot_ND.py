import pycxsom as cx
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from inputs_ND import *


if __name__ == '__main__':
    I =Inputs(sys.argv[1])
    P = GeoND(inputs=I)
    I.plot()
