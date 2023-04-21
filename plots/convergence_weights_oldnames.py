import pycxsom as cx
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


#directories = '../../xp_list'
directories = '../xp_cercle'
n = 10
xps = [f'cercle{i}' for i in range(n)]
#xps = ['2SOM_SU_000', '2SOM_SU_003','2SOM_SU_004']
n = len(xps)
maps = ['M1','M2']

k = 100
begin = 0
end = 10000
timesteps = range(begin+k,end,k)
fig,axes = plt.subplots(1,len(maps))

map_size_0 = 500
map_size_1 = 100

plt.rc('text', usetex=True)

def varpath(name,timeline):
    return os.path.join(timeline, name)

def norm(value):
    if len(value)>1:
        return np.sqrt(value[0]**2 + value[1]**2)
    else:
        return abs(value)

for i,m in enumerate(maps):
    print(m)
    diff_tab = np.empty(n, dtype='object')
    for xpi, xp in enumerate(xps):
        print(xp)
        weight_dict = dict()
        path_in = os.path.join(directories, xp, f'wgt')
        weights = os.listdir(os.path.join(path_in,m))
        for w in weights:
            #parcours des timesteps
            diff = []
            wfk = cx.variable.Realize(varpath(w,os.path.join(path_in,m)))
            wfk.open()
            for t in timesteps:
                wf_t = np.array(wfk[t])
                wf_p = np.array(wfk[t-k])
                difference = np.array([wf_t[ii] - wf_p[ii] for ii in range(len(wf_t))])
                #difference = np.sort(difference)
                d = np.mean(difference)
                diff.append(d)

            wfk.close()
            #fin parcours timesteps
            weight_dict[w] = np.array(diff)

        #fin parcours weights
        diff_tab[xpi] = weight_dict

    #fin parcours xps
    #pour We, Wc d'une carte m
    for wname in diff_tab[0].keys():
        print(wname)
        #moyenne
        diff_moy = np.array(diff_tab[0][wname])
        for wdict in diff_tab[1:]:
            diff_moy = np.add(diff_moy, (1/n)*wdict[wname])

        #variance
        diff_var = np.array(np.square(diff_tab[0][wname]- diff_moy))
        for wdict in diff_tab[1:]:
            diff_var = np.add(diff_var, np.square(wdict[wname] - diff_moy))

        diff_var =np.sqrt(diff_var/n)

        axes[i].plot(timesteps,diff_moy, label = f'{wname}' )
        axes[i].set_title('$M^{(%d)}$'%(i+1))
        axes[i].fill_between(timesteps, diff_moy + diff_var, diff_moy - diff_var, alpha = 0.7)
        axes[i].legend()


plt.show()
