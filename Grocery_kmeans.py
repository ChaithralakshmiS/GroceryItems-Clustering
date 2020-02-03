#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import re
from collections import Counter

from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,MeanShift,AgglomerativeClustering
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
from matplotlib import style
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import dendrogram, linkage

from mpl_toolkits.mplot3d import Axes3D


# In[2]:


#Read files
item=pd.read_csv('data/item_to_id.csv',index_col='Item_id')
purchase_original=pd.read_csv('data/purchase_history.csv')
item.replace(u'\xa0',u'', regex=True, inplace=True)

#purchase.head()
#purchase.user_id.value_counts()





# In[3]:


#For each user, get item id and count
def item_counts(same_user_df):
    all_item = same_user_df['id'].str.split(',').sum()
    return pd.Series(Counter(int(id) 
    for id in all_item))

def find_user_item_counts(purchase_original):
    purchase=purchase_original
    user_item_counts = purchase.groupby("user_id").apply(item_counts).unstack(fill_value=0)
    #user_item_counts.head()
    return user_item_counts

def normalize_user_item_count(user_item_counts):
    #Normalize user_item_counts 
    item_norm = normalize(user_item_counts.values, axis=0)
    item_item_similarity = item_norm.T.dot(item_norm)
    item_item_similarity = pd.DataFrame(item_item_similarity,index=user_item_counts.columns,columns=user_item_counts.columns)
    return item_item_similarity

def dimensionality_reduction_pca(user_item_counts,item_item_similarity,dimensions_in):
    #dimension reduction
    dimensions=dimensions_in
    pca = PCA(n_components=dimensions)
    items_pca = pca.fit_transform(item_item_similarity)
    items_pca = pd.DataFrame(items_pca,index=user_item_counts.columns,columns=["pca{}".format(index) for index in range(dimensions)])
    return pca,items_pca


# In[4]:


def kmeans_cluster(pca,items_pca,n_clusters,n_components=4):
    kmeans = KMeans(init='k-means++',n_clusters=n_clusters)
    kmeans.fit(items_pca.values[:, :n_components])
    return kmeans.inertia_

def kmeans_clustering_algo(purchase_original,dimensions_list,min_k,max_k):
    for i in range(len(dimensions_list)):
        dimensions=dimensions_list[i]
        inertia=list()
        print('*************************************************************************')
        print('*********************** PCA components = ',dimensions,'*****************************')
        for k in range (min_k, max_k):
            user_item_counts=find_user_item_counts(purchase_original)
            item_item_similarity=normalize_user_item_count(user_item_counts)
            #print('item_item_similarity = ',pd.DataFrame(item_item_similarity).shape)
            pca,items_pca=dimensionality_reduction_pca(user_item_counts,item_item_similarity,dimensions)
            explained_variance_by_k = pca.explained_variance_ratio_.cumsum()
            inertia.append(kmeans_cluster(pca,items_pca,n_clusters=k,n_components=dimensions))
            ks=range(1,k+1)
        print('Variance explained = {:.2f}%'.format(100 * sum(pca.explained_variance_ratio_[:dimensions])))

        print('Inertia = ',inertia)
        fig = plt.figure(figsize=(10,4))

        plt.plot(ks, inertia,marker='o')
        plt.show()
        print('\n\n\n\n')


# In[5]:


def show_clusters(items_pca,labels):

    fig = plt.figure(figsize=(15, 15))

    colors =  itertools.cycle (["b","g","r","c","m","y","k"])
    color_names = ['blue','green','red','cyan','magenta','yellow','black']
    grps = items_pca.groupby(labels)
    print('\n')

    for label,grp in grps:
        
        print("*************** Cluster_no = ",label+1,' ******* color = ',color_names[label],'   *****************')
        plt.scatter(grp.pca1,grp.pca2,c=next(colors),label = label)

        names = item.loc[ grp.index,"Item_name"]
        cluster_groups=[]
        for index, name in enumerate(names):
            name_count=index+1
            cluster_groups.append(str(name_count)+'. '+name)
        print(cluster_groups)
        print('\n')
    for itemid in items_pca.index:
        x = items_pca.loc[itemid,"pca1"]
        y = items_pca.loc[itemid,"pca2"]
        name = item.loc[itemid,"Item_name"]
        name = re.sub('\W', ' ', name)

        plt.text(x,y,name)


# In[6]:


# Clustering
def cluster(purchase_original,k,dimensions=10):

    n_components=dimensions
    inertia=list()

    user_item_counts=find_user_item_counts(purchase_original)
    item_item_similarity=normalize_user_item_count(user_item_counts)
    pca,items_pca=dimensionality_reduction_pca(user_item_counts,item_item_similarity,dimensions)
    explained_variance_by_k = pca.explained_variance_ratio_.cumsum()
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(items_pca.values[:, :n_components])
    inertia.append(kmeans_cluster(pca,items_pca,n_clusters=k,n_components=dimensions))
    ks=range(1,k+1)
    print('Variance explained = {:.2f}%'.format(100 * sum(pca.explained_variance_ratio_[:dimensions])))
    show_clusters(items_pca, kmeans.labels_)


# In[7]:


def hierarchical_clustering(purchase_original,dimensions):
    user_item_counts=find_user_item_counts(purchase_original)
    item_item_similarity=normalize_user_item_count(user_item_counts)
    pca,items_pca=dimensionality_reduction_pca(user_item_counts,item_item_similarity,dimensions)
    plt.figure(figsize=(10, 7))  
    plt.title("Customer Dendograms")  
    
    dend = shc.dendrogram(shc.linkage(items_pca, method='ward'))  
    cluster = AgglomerativeClustering(n_clusters= 7, affinity='euclidean', linkage='ward')  
    cluster.fit_predict(items_pca)  


    plt.figure(figsize=(15, 15))  
    show_clusters(items_pca, cluster.labels_)


# In[8]:


def k_means_execution():
    item=pd.read_csv('data/item_to_id.csv',index_col='Item_id')
    purchase_original=pd.read_csv('data/purchase_history.csv')
    item.replace(u'\xa0',u'', regex=True, inplace=True)
    kmeans_clustering_algo(purchase_original,dimensions_list=[5,6,7,9,10,11,15,30],min_k=1,max_k=20)
    x=cluster(purchase_original,k=7,dimensions=10)


# In[9]:


def hierarchical_execution():
    item=pd.read_csv('data/item_to_id.csv',index_col='Item_id')
    purchase_original=pd.read_csv('data/purchase_history.csv')
    item.replace(u'\xa0',u'', regex=True, inplace=True)
    hierarchical_clustering(purchase_original,dimensions=10)


# In[10]:


k_means_execution()
hierarchical_execution()


# In[ ]:




