import pycxsom as cx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.image as mpimg
import os
import sys
from correlation_ratio import *
MAP_SIZE = 100
from PIL import Image
import colorsys
import math

def colorwheel():
    d = (int(MAP_SIZE*np.sqrt(2)))
    im = np.zeros((d,d,3))
    radius = d/2.0
    cx, cy = d/2, d/2
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            rx = x - cx
            ry = x - cy
            s = (rx ** 2.0 + ry ** 2.0) ** 0.5 / radius
            if s <= 1.0:
                h = ((math.atan2(ry, rx) / math.pi) + 1.0) / 2.0
                rgb = colorsys.hsv_to_rgb(h, s, 1.0)
                im[x,y,:] = [int(c*255) for c in rgb]

    coins = int((d-MAP_SIZE)/2)
    im2 = im[coins:coins+MAP_SIZE,coins:coins+MAP_SIZE,:]
    im2 = im2.astype(int)
    return im2

def colorwheel_f(couple):
    i = int(MAP_SIZE*couple[0])
    j = int(MAP_SIZE*couple[1])
    d = (int(MAP_SIZE*np.sqrt(2)))
    im = np.zeros((d,d,3))
    radius = d/2.0
    cx, cy = d/2, d/2
    rx = i - cx
    ry = j - cy
    s = (rx ** 2.0 + ry ** 2.0) ** 0.5 / radius
    h = ((math.atan2(ry, rx) / math.pi) + 1.0) / 2.0
    rgb = colorsys.hsv_to_rgb(h, s, 1.0)
    tuple = np.array([int(c*255) for c in rgb])
    return  tuple.astype(int)

def colormap(type='bremm'):
    __location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
    file = os.path.join(__location__, 'bremm.png')
    if type=='ziegler':
        file = os.path.join(__location__, 'ziegler.png')
    if type=='teul':
        file = os.path.join(__location__, 'teulingfig2.png')

    image = mpimg.imread(file)
    return image

def colormap_a(weights,type):
    imagec = np.zeros((MAP_SIZE,MAP_SIZE,3))
    cmap = colormap(type)
    N = cmap.shape[0]
    M = cmap.shape[1]
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            imagec[i,j,:] = cmap[int(weights[i,j,0]*N), int(weights[i,j,1]*M)]
    return imagec

def colormap_l(liste,type,n):
    imagec = np.zeros((n, 3))
    cmap = colormap(type)
    N = cmap.shape[0]
    M = cmap.shape[1]
    for i in range(n):
        imagec[i,:] = cmap[int(liste[i,1]*(N-1)), int(liste[i,0]*(M-1))]
    return imagec
