import pyfpgrowth
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
import time
import re
import networkx as nx
import pickle 
import matplotlib.pyplot as plt
import community.community_louvain as community

def fp_growth():
    with open("fpgrowth.txt", "rb") as myFile:
        result = pickle.load(myFile)

    G = nx.Graph(result)
    G.add_nodes_from(result.keys())

    fig, axs = plt.subplots(1,1, figsize=(50,50))
    pos = nx.circular_layout(G)
    nx.draw(G,axis=axs, pos=pos, node_size=1, with_labels=True)
    #plt.figure(2,figsize=(33, 33))

    plt.show()

    plt.figure(2,figsize=(50, 50))
    nx.draw(G, with_labels=True)
    plt.show()


    #Communities
    partition = community.best_partition(G)


    #
    print("-----Communities-----")
    partition = community.best_partition(G)  

    pos = nx.spring_layout(G)  
    plt.figure(figsize=(50, 50))
    plt.axis('off')
    nx.draw_networkx_nodes(G, pos, node_size=100, cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.show(G)

    mod = community.modularity(partition,G)
    print("modularity:", mod)

    node_size = []

    # first community against the others
    for node, cmty in partition.items():
        if cmty == 1:
            node_size.append(900)
        else:
            partition[node] = 0  # I put all the other communities in one communitiy
            node_size.append(300)

    plt.figure(figsize=(50, 50))
    plt.axis('off')
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, cmap=plt.cm.winter, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.show(G)

    #Connected components
    largest_cc = max(nx.connected_components(G), key=len)
    #print(len(largest_cc))

    print("The Recommended Clusters based off of connected components:")
    for item in largest_cc:
        print (item)

    # Clustering coefficients
    print('-----Clustering Coefficients------')
    print(nx.clustering(G))

    print("--------Clustering based off of Communities------")
    communities_generator = nx.algorithms.community.centrality.girvan_newman(G)
    top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    print(next_level_communities)
#     x = (sorted(map(sorted, next_level_communities)))
#     print(x)




