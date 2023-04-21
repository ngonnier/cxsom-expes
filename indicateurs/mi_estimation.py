from mi_evolution import mutual_information,get_inputs,get_bmus
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from os.path import isfile, join
import pandas as pd
import pycxsom as cx
import os


nxp = 12
stepf = 200
nf = 10000
test_times = list(range(0,200,5))+list(range(200,1000,200))
