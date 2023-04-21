import sys
import pycxsom as cx
import numpy as np
import tkinter as tk
import matplotlib as plt
from distortion_maps import *
import re
import networkx as nx
import os

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]



class DistoViewer(cx.tkviewer.At):
    def __init__(self, master, title, varpaths, map_names, map_to_plot, films_numbers,figsize=(4, 5), dpi=100):
        super().__init__(master, title, figsize, dpi)
        self.varpaths = varpaths
        self.map_names = map_names
        self.map_to_plot = map_names[map_to_plot]
        self.ax = self.fig.add_subplot(111,projection='3d')
        self.films_numbers = films_numbers
    # This is the inherited (and overrided) method for drawing
    def on_draw_at(self, at):
        #trouver le step le plus proche
        test_number = find_nearest(self.films_numbers,at)
        df  =read_data("test-film-"+str(test_number),test_number,self.varpaths,self.map_names)
        df_sort = trie(df,'bmu'+self.map_to_plot)
        #df_sorty = trie(df,'bmu'+map_names[1])
        #df_sortz = trie(df,'bmu'+map_names[2])
        self.ax.clear()    # self.fig is the matplotlib figure
        ax = self.fig.gca()
        self.ax.plot(df_sort['we'+self.map_names[0]],df_sort['we'+self.map_names[1]],df_sort['we'+self.map_names[2]])
        self.ax.scatter(df_sort['we'+self.map_names[0]],df_sort['we'+self.map_names[1]],df_sort['we'+self.map_names[2]],c = df_sort[f'bmu{self.map_to_plot}'],cmap = 'plasma')
        self.ax.set_title(f'Map {map}')
        self.ax.set_xlabel(self.map_names[0])
        self.ax.set_ylabel(self.map_names[1])
        self.ax.set_zlabel(self.map_names[2])



class WeightViewer(cx.tkviewer.At):
    def __init__(self, master, title, varpaths, inputs, films_numbers, root_dir, figsize=(5,5), dpi=100):
        super().__init__(master, title, figsize, dpi)
        self.varpaths = varpaths
        self.map = varpaths[0].split(os.sep)[-2]
        self.films_numbers = films_numbers
        self.mdir = root_dir
        self.inputs= inputs

    # This is the inherited (and overrided) method for drawing
    def on_draw_at(self, at):
        #test_number = find_nearest(self.films_numbers,at)
        #df  =read_data("test-"+str(test_number),test_number,self.varpaths,self.map_names)
        self.fig.clear()    # self.fig is the matplotlib figure
        ax = self.fig.gca()
        test_number = find_nearest(self.films_numbers,at)
        varpath_bmu = os.path.join(self.mdir,f'rlx-test-{test_number}',self.map,'BMU.var')
        with cx.variable.Realize(varpath_bmu) as fbmu:
            r = fbmu.time_range()
            bmus = np.zeros(r[1],dtype=object)
            for i in range(r[1]):
                bmus[i] = fbmu[i]
            ax.scatter(bmus,self.inputs['I1'],c='r')
            ax.scatter(bmus,self.inputs['I2'],c='b')
            #ax.scatter(bmus,self.inputs['I3'],c='g')

        nb_curves = 0
        for varpath in self.varpaths:
            #print(varpath)
            mname = varpath.split(os.sep)[-2]
            _, timeline, name = cx.variable.names_from(varpath)
            with cx.variable.Realize(varpath) as v:
                try:
                    Y = v[at]
                    X = np.linspace(0,1,len(Y))
                    ax.plot(X, Y, label='({}){}'.format(timeline, name),alpha=0.5)
                    nb_curves += 1
                except cx.error.Busy:
                    pass
                except cx.error.Forgotten:
                    pass
            if nb_curves > 0:
                ax.legend()



class GraphViewer(cx.tkviewer.At):
    def __init__(self, master, title, varpaths,map_names,films_numbers,figsize=(8, 5), dpi=100):
        super().__init__(master, title, figsize, dpi)
        self.varpaths = varpaths
        self.map_names = map_names
        self.films_numbers = films_numbers
        self.graph = nx.Graph()

        #tracer les cartes et connecter les cartes les unes au sein des autres
        colors = ['red','blue','green','yellow','grey','black']
        for j,m in enumerate(self.map_names):
            self.graph.add_nodes_from([((m,n),{'color':colors[j],'pos':(n,j)}) for n in range(1,map_size+1)])

    # This is the inherited (and overrided) method for drawing
    def on_draw_at(self, at):
        #trouver le step le plus proche
        test_number = find_nearest(self.films_numbers,at)
        df  =read_data("test-film-"+str(test_number),test_number,self.varpaths,self.map_names)
        bmus = df[['bmu'+m for m in self.map_names]]
        map_names_pairs = [(self.map_names[i],self.map_names[j]) for i in range(len(self.map_names)) for j in range(i+1,len(self.map_names))]
        self.graph.remove_edges_from(self.graph.edges())
        for idx,row in bmus.iterrows():
            #row format : dict - attributes: name, dtype, bmuM1,bmuM2,bmuM3
            for m1,m2 in map_names_pairs:
                val1 = math.floor(map_size*row[f'bmu{m1}'])
                val2= math.floor(map_size*row[f'bmu{m2}'])
                self.graph.add_edge((m1,val1),(m2,val2))

        values = nx.get_node_attributes(self.graph,'color')
        cols = [values[node] for node in self.graph.nodes()]
        pos=nx.get_node_attributes(self.graph,'pos')

        self.fig.clear()    # self.fig is the matplotlib figure
        ax = self.fig.gca()
        nx.draw(self.graph,pos,node_color=cols,node_size=5)
