import sys
import pycxsom as cx
import numpy as np
import tkinter as tk
import matplotlib as plt
from distortion_maps import *
import re
from viewers import *


if len(sys.argv) < 2:
    print()
    print('Usage:')
    print('  {} <directory>'.format(sys.argv[0]))
    print()
    sys.exit(0)

# Let us check that all the variables have a 1D shape.
varpaths = sys.argv[1]
error = False
# Let us launch a graphical interface

# This is the main frame
root = tk.Tk()

#Time slider associated with tests numbers.
files = os.listdir(varpaths)
films_numbers = np.array([int(re.search(r"\d+",s).group(0)) for s in files if re.match(r"^init-test-\d+$",s)])
films_numbers = np.sort(films_numbers)
min = np.min(films_numbers)
max = np.max(films_numbers)
#min = 9000
#max = 1000

# We add an instance of our viewer in the GUI.
def w_varpaths(dir,map):
    mdir = os.path.join(dir,'wgt', map)
    files = os.listdir(mdir)
    print(files)
    return [os.path.join(mdir,f) for f in files]

def bmu_varpath(dir,map,num):
    mdir = os.path.join(dir,'wgt', map)
    return os.path.join(dir,f'rlx-test-{num}',map,'BMU.var')

def inp_path(dir):
    mdir= os.path.join(dir,"input-test")
    files = os.listdir(mdir)
    return [os.path.join(mdir,f) for f in files]

weightframe = tk.Frame(root)
weightframe.pack(side = tk.BOTTOM)


slider = cx.tkviewer.HistorySlider(root,'test numbers',min,max,min)

slider.widget().pack(fill=tk.BOTH, side=tk.TOP)


inputs = dict()
for f in inp_path(varpaths):
    with cx.variable.Realize(f) as inp:
        r = inp.time_range()
        inps = np.zeros(r[1],dtype=object)
        for i in range(r[1]):
            inps[i] = inp[i]
        name = f.split(os.sep)[-1]
        inputs[name[:-4]] = inps


viewer4 = WeightViewer(weightframe, 'Map 1 Weights', w_varpaths(varpaths,'M1'),inputs, films_numbers, varpaths)
viewer4.widget().pack(side=tk.LEFT)
viewer4.set_history_slider(slider) # This viewer is controlled by our slider.

viewer5 = WeightViewer(weightframe, 'Map 2 Weights',  w_varpaths(varpaths,'M2'),inputs, films_numbers,varpaths)
viewer5.widget().pack(side=tk.LEFT)
viewer5.set_history_slider(slider) # This viewer is controlled by our slider.

# viewer6 = WeightViewer(weightframe, 'Map 3 Weights',  w_varpaths(varpaths,'M3'),inputs, films_numbers,varpaths)
# viewer6.widget().pack(side=tk.LEFT)
# viewer6.set_history_slider(slider) # This viewer is controlled by our slider.

# Then we start the GUI.
tk.mainloop()
