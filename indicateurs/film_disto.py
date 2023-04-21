import sys
import pycxsom as cx
import numpy as np
import tkinter as tk
import matplotlib as plt
from distortion_maps import *
import re
from viewers import *


if len(sys.argv) < 3:
    print()
    print('Usage:')
    print('  {} <directory> <maps> '.format(sys.argv[0]))
    print()
    sys.exit(0)

# Let us check that all the variables have a 1D shape.
varpaths = sys.argv[1]
map_names = sys.argv[2:]
error = False
# Let us launch a graphical interface

# This is the main frame
root = tk.Tk()
#Time slider associated with tests numbers.

files = os.listdir(varpaths)
films_numbers = np.array([int(re.search(r"\d+",s).group(0)) for s in files if re.match(r"^init-test-film-\d+$",s)])
films_numbers = np.sort(films_numbers)
min = np.min(films_numbers)
max = np.max(films_numbers)
#min = 9000
#max = 9100

slider = cx.tkviewer.HistorySlider(root,'test numbers',min,max,min)

slider.widget().pack(fill=tk.BOTH, side=tk.TOP)


# We add an instance of our viewer in the GUI.
distoframe = tk.Frame(root,height=5)
distoframe.pack(side = tk.TOP, fill = tk.BOTH)
viewer1 = DistoViewer(root, 'Distortion map', varpaths, map_names,0,films_numbers)
viewer1.widget().pack(side=tk.LEFT)
viewer1.set_history_slider(slider) # This viewer is controlled by our slider.

viewer2 = DistoViewer(root, 'Distortion map', varpaths, map_names,1,films_numbers)
viewer2.widget().pack(side=tk.LEFT)
viewer2.set_history_slider(slider) # This viewer is controlled by our slider.

viewer3 = DistoViewer(root, 'Distortion map', varpaths, map_names,2,films_numbers)
viewer3.widget().pack(side=tk.LEFT)
viewer3.set_history_slider(slider) # This viewer is controlled by our slider.


"""
# We add an instance of our viewer in the GUI.
def w_varpaths(dir,map):
    mdir = os.path.join(dir,'wgt', map)
    files = os.listdir(mdir)
    print(files)
    return [os.path.join(mdir,f) for f in files]

weightframe = tk.Frame(root)
weightframe.pack(side = tk.BOTTOM)

viewer4 = WeightViewer(weightframe, 'Map 1 Weights', w_varpaths(varpaths,'M1'))
viewer4.widget().pack(side=tk.LEFT)
viewer4.set_history_slider(slider) # This viewer is controlled by our slider.

viewer5 = WeightViewer(weightframe, 'Map 2 Weights',  w_varpaths(varpaths,'M2'))
viewer5.widget().pack(side=tk.LEFT)
viewer5.set_history_slider(slider) # This viewer is controlled by our slider.

viewer6 = WeightViewer(weightframe, 'Map 3 Weights',  w_varpaths(varpaths,'M3'))
viewer6.widget().pack(side=tk.LEFT)
viewer6.set_history_slider(slider) # This viewer is controlled by our slider.
"""
# Then we start the GUI.
tk.mainloop()
