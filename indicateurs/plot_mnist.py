import pycxsom as cx
import numpy as np
import os
import matplotlib.pyplot as plt
import math

def varpath(name,timeline):
    return os.path.join(timeline, name)

path = 'multisom/mnist/wgt/M8/We.var'

num = 50000
w_e= cx.variable.Realize(path)
w_e.open()


#mettre tous les poids dans une image, taille carte 10x10 : montre une carte
map_size = 10
patch_size = 7
image_size = 28
w = w_e[num]
w_e.close()

image = np.zeros((map_size*patch_size,map_size*patch_size))
for ii in range(map_size):
    for jj in range(map_size):
        weight = w[ii,jj]
        #print(weight.shape)
        weight = weight.reshape((patch_size,patch_size))
        image[patch_size*ii:patch_size*(ii+1),patch_size*jj:patch_size*(jj+1)] = weight
"""
plt.figure()
plt.vlines(range(0,70,7),0,70,colors='b')
plt.hlines(range(0,70,7),0,70,colors='b')
plt.imshow(image)
plt.colorbar()
plt.show()
"""
#tests :

def pos2index(p0,p1):
    i = math.floor(map_size*p0)
    j = math.floor(map_size*p1)
    if(i == 10):
        i = 9
    if(j==10):
        j=9
    return i,j

wbmus = dict()
inputs = dict()

for i in range(1,17):
    path_t = f'multisom/mnist/rlx-test-55000-0/M{i}/BMU.var'
    path_w = f'multisom/mnist/wgt/M{i}/We.var'
    path_in = f'multisom/mnist/input-test/I{i}.var'

    w_e= cx.variable.Realize(path_w)
    w_e.open()
    w = w_e[55000]
    w_e.close()

    BMUf= cx.variable.Realize(path_t)
    BMUf.open()
    r = BMUf.time_range()
    bmu = np.array([BMUf[at] for at in range(r[0],r[1])])
    BMUf.close()

    inpf= cx.variable.Realize(path_in)
    inpf.open()
    r = inpf.time_range()
    input = np.array([inpf[at] for at in range(r[0],r[1])])
    inpf.close()

    #w : tableau 10 * 10 * 49 valeurs
    #bmu : 1000 positions array(2)
    wbmu = np.array([w[pos2index(b[0],b[1])] for b in bmu])
    #print(wbmu.shape)
    #print(input.shape)
    wbmus[i] = wbmu
    inputs[i] = input

#print one random test as an image
numtest = np.random.randint(0,9990)
print(numtest)
image_test = np.zeros((image_size,image_size))
image_input = np.zeros((image_size,image_size))

ii = 0
jj = 0

for m in range(1,17):
    id = m-1
    ii = id // 4
    jj = id % 4

    bmuw = wbmus[m][numtest]
    inp = inputs[m][numtest]

    #print(weight.shape)
    bmuw = bmuw.reshape((patch_size,patch_size))
    inp= inp.reshape((patch_size,patch_size))

    #image 4 * 7 par 4 * 7
    #bmuw : liste de 16 elements de 7
    image_test[patch_size*ii:patch_size*(ii+1),patch_size*jj:patch_size*(jj+1)] = bmuw
    image_input[patch_size*ii:patch_size*(ii+1),patch_size*jj:patch_size*(jj+1)] = inp


plt.figure()
plt.vlines(range(0,70,7),0,70,colors='b')
plt.hlines(range(0,70,7),0,70,colors='b')
plt.imshow(image_input)
plt.colorbar()
plt.figure()
plt.vlines(range(0,70,7),0,70,colors='b')
plt.hlines(range(0,70,7),0,70,colors='b')
plt.imshow(image_test)
plt.colorbar()
plt.show()
