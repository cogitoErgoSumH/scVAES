# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")


import numpy as np
import matplotlib.pyplot as plt
from vasc import vasc
from config import config
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI

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


if __name__ == '__main__':

    file_name = '../datasets/pbmc68k_filtered_cell.csv'
    label_name = '../datasets/pbmc68k_filtered_cell_labels.csv'
    
    PREFIX = file_name
    expr = pd.read_csv(file_name,sep=',',index_col=0).values
    label = pd.read_csv(label_name, sep=',', index_col=0)['cell_type'].values
    # label = pd.read_csv(label_name, sep=',', index_col=0, header=None).values.reshape(56,) ###
    # label = pd.Categorical(label).codes ###
    
    n_cell,_ = expr.shape
    if n_cell > 150:
        batch_size=config['batch_size']
    else:
        batch_size=32 

    res = vasc( expr,var=False,
                epoch=config['epoch'],
                latent=config['latent'],
                annealing=False,
                batch_size=batch_size,
                #prefix=PREFIX,
                label=label,
                scale=config['scale'],
                patience=config['patience'] 
            )
    
    tsne = show_tSNE(res, label, return_tSNE=True)