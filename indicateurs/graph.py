import networkx as nx
from plot_func import get_tests
from distortion_maps import read_data
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import math
map_size = 500

if __name__ == "__main__":
    if len(sys.argv)<4:
            sys.exit("Usage: python3 distortion_maps.py <path_to_dir> <test_number> <n_samp> <map names")
    else:
        dir = sys.argv[1]
        n_samp = int(sys.argv[3])
        test_name = sys.argv[2]
        map_names = sys.argv[4:]

    G = nx.Graph()

    #tracer les cartes et connecter les cartes les unes au sein des autres
    colors = ['red','blue','green','yellow','grey','black']
    for j,m in enumerate(map_names):
        G.add_nodes_from([((m,n),{'color':colors[j],'pos':(n,j)}) for n in range(1,map_size+1)])
        G.add_edges_from([((m,i),(m,i+1)) for i in range(1,map_size)])


    #lis les données tests sous forme de DataFrame pandas
    test_numbers_0 = int(test_name)
    df  =read_data("test-"+str(test_numbers_0)+"-0",test_numbers_0,dir,map_names)

    #ajouter les connexions entre noeuds sur les entrées de tests : n1 et n2 connectés s'ils s'activent en même temps.
    #option arête unique

    bmus = df[['bmu'+m for m in map_names]]
    print(bmus)
    map_names_pairs = [(map_names[i],map_names[j]) for i in range(len(map_names)) for j in range(i+1,len(map_names))]
    print(map_names_pairs)
    for idx,row in bmus.iterrows():

        if idx>0:
            #row format : dict - attributes: name, dtype, bmuM1,bmuM2,bmuM3
            for m1,m2 in map_names_pairs:
                val1 = math.floor(map_size*row[f'bmu{m1}'])
                val2= math.floor(map_size*row[f'bmu{m2}'])
                print(val1)
                G.add_edge((m1,val1),(m2,val2))



    values = nx.get_node_attributes(G,'color')
    cols = [values[node] for node in G.nodes()]
    pos=nx.get_node_attributes(G,'pos')
    plt.figure()
    nx.draw(G,pos,node_color=cols,node_size=5)
    plt.show()
