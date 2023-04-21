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
viewer1 = GraphViewer(root, 'Distortion map', varpaths, map_names,films_numbers)
viewer1.widget().pack(side=tk.LEFT)
viewer1.set_history_slider(slider) # This viewer is controlled by our slider.

# Then we start the GUI.
tk.mainloop()
