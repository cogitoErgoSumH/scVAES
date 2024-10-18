# -*- coding: utf-8 -*-
"""
Created on Fri May 14 00:06:51 2021

@author: Administrator
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

X = pd.read_csv("./pbmc68k_filtered_cell.csv", index_col=0)
expression_data = np.array(X, dtype=np.int)
clusters = pd.read_csv("./pbmc68k_filtered_cell_labels.csv", index_col=0)
cell_type = np.array(clusters['cell_type'])

my_pca = PCA(n_components=10)
res_pca = my_pca.fit_transform(X)



def show_tSNE(latent, labels, return_tSNE=False):
    
    if latent.shape[1] != 2:
        latent = TSNE().fit_transform(latent)
    
    latent_df = pd.DataFrame(latent, columns=['tsne1','tsne2'])
    latent_df['labels'] = labels
    unique_label = np.unique(labels)
    colors=['grey','gold','darkviolet','turquoise','r','g','salmon', 'c','b','m','y','darkorange','lightgreen','plum', 'tan','khaki', 'pink', 'skyblue','lawngreen','k']
    
    labels_pred = KMeans(n_clusters=len(unique_label)).fit_predict(latent)
    print("silhouette_score =", silhouette_score(latent, labels))
    print("ARI =", ARI(labels, labels_pred))
    print("NMI =", NMI(labels, labels_pred))
    
    plt.figure(figsize=(10, 8))
    for i, v in enumerate(unique_label):
        plt.scatter(latent_df['tsne1'][latent_df['labels']==v], latent_df['tsne2'][latent_df['labels']==v],
                    label=v, c=colors[i], edgecolors='none', s=20)
    plt.legend(prop = {'size':14}, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    
    #plt.figure(figsize=(8, 8)) 
    #plt.scatter(latent[:, 0], latent[:, 1], c=labels, cmap=cmap, edgecolors='none')
    plt.axis("off")
    plt.tight_layout()

    if return_tSNE:
        return latent
    

tsne = show_tSNE(res_pca, cell_type, return_tSNE=True)