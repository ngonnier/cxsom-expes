import pycxsom as cx
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


#directories = '../../xp_list'
directories = '../xp_cercle'
n = 10
xps = [f'cercle{i}' for i in range(n)]
#xps = ['2SOM_SU_000', '2SOM_SU_001','2SOM_SU_003','2SOM_SU_004']
n = len(xps)
maps = ['M1','M2']

k = 10000
begin = 10000
end = 500000
timesteps = range(begin,500000,k)
fig,axes = plt.subplots(len(maps),2)



map_size_0 = 100
map_size_1 = 100

plt.rc('text', usetex=True)

def varpath(name,timeline):
    return os.path.join(timeline, name)

def norm(value):
    if len(value)>1:
        return np.sqrt(value[0]**2 + value[1]**2)
    else:
        return abs(value)

def get_diff(timesteps,k,w,xp):
    #parcours des timesteps
    diff = []
    print(w)
    for t in timesteps[1:]:
        print(t)
        pathprev = os.path.join(directories, xp, f'zfrz-{t-k}-wgt')
        wfprev = cx.variable.Realize(varpath(w,os.path.join(pathprev,m)))
        path = os.path.join(directories, xp, f'zfrz-{t}-wgt')
        wfk = cx.variable.Realize(varpath(w,os.path.join(path,m)))
        wfk.open()
        wfprev.open()
        wf_t = np.array(wfk[0])
        wf_p = np.array(wfprev[0])

        wf_t= wf_t.reshape(10000,2)
        wf_p= wf_p.reshape(10000,2)
        #pour chaque unité : |w(t) - w(t-k)|
        difference = np.array([norm(wf_t[ii] - wf_p[ii]) for ii in range(len(wf_t))])
        #difference = np.sort(difference)
        d = np.median(difference)
        diff.append(d)
        wfk.close()
        wfprev.close()
    return np.array(diff)

for i,m in enumerate(maps):
    print(m)
    diff_tab = np.empty(n, dtype='object')

    #xps : r_c = 0.02
    for xpi, xp in enumerate(xps):
        print(xp)
        weight_dict = dict()
        path_in = os.path.join(directories, xp, f'zfrz-{k}-wgt')
        weights = os.listdir(os.path.join(path_in,m))
        for w in weights:
            weight_dict[w] = get_diff(timesteps,k,w,xp)

        axes[i,0].plot(timesteps[1:], weight_dict['We-0.var'], label = f'$r_c = 0.02$')
        axes[i,1].plot(timesteps[1:], weight_dict['Wc-0.var'], label = f'$r_c = 0.02$')
        axes[i,0].set_title(f'$\\omega_e, M^{i}$')
        axes[i,1].set_title(f'$\\omega_c, M^{i}$')
        #axes[i].fill_between(timesteps, diff_moy + diff_var, diff_moy - diff_var, alpha = 0.7)

    #get values for xp rc = 0.05
    weight_dict = dict()
    path_in = os.path.join(directories, '2SOM_S_003', f'zfrz-1000-wgt')
    weights = os.listdir(os.path.join(path_in,m))
    timesteps_005 = range(1000,500000,10000)
    for w in weights:
        weight_dict[w] = get_diff(timesteps_005,10000, w,'2SOM_S_003')

    axes[i,0].plot(timesteps_005[1:], weight_dict['We-0.var'], label = f'$r_c = 0.05$')
    axes[i,1].plot(timesteps_005[1:], weight_dict['Wc-0.var'], label = f'$r_c = 0.05$')
    axes[i,0].set_title(f'$\\omega_e, M^{i}$')
    axes[i,1].set_title(f'$\\omega_c, M^{i}$')

    #get values for xp rc = 0.03
    weight_dict = dict()
    path_in = os.path.join(directories, '2SOM_S_004', f'zfrz-10000-wgt')
    weights = os.listdir(os.path.join(path_in,m))
    timesteps_003 = range(10000,500000,10000)
    for w in weights:
        weight_dict[w] = get_diff(timesteps_003,10000, w,'2SOM_S_004')

    axes[i,0].plot(timesteps_003[1:], weight_dict['We-0.var'], label = f'$r_c = 0.03$')
    axes[i,1].plot(timesteps_003[1:], weight_dict['Wc-0.var'], label = f'$r_c = 0.03$')
    axes[i,0].set_title(f'$\\omega_e, M^{i}$')
    axes[i,1].set_title(f'$\\omega_c, M^{i}$')



    axes[i,0].legend()
    axes[i,1].legend()


plt.show()
