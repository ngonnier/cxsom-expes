import os
import time
import numpy as np
import pycxsom as cx
import matplotlib as mpl
import matplotlib.pyplot as plt

MAX_DURATION = 2000

X_We_path = os.path.join('data','wgt', 'X', 'We.var')
X_Wc_path = os.path.join('data','wgt', 'X', 'Wc-0.var')
Y_We_path = os.path.join('data','wgt', 'Y', 'We.var')
Y_Wc_path = os.path.join('data','wgt', 'Y', 'Wc-0.var')


# Here, we access xsom data to fill the plots.
with cx.variable.Bind(X_We_path) as X:
    with cx.variable.Bind(X_Wc_path) as Y:
        we = X[MAX_DURATION]
        wc = Y[MAX_DURATION]
        plt.plot(range(500),we)
        plt.plot(range(500),wc)

plt.show()
