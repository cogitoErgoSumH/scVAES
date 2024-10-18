#!/usr/bin/env python
# coding: utf-8

# # loading modules

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np

#from model import scVI_final as scVI
import scVI
import pandas as pd


import time

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

# 训练，从helper复制过来的
def train_model(model, expression, sess, num_epochs, step=None, batch=None, kl=None):
    
    expression_train, expression_test = expression
    
    scVI_batch = batch is not None # 是否在数据源中加入批次，代码有两次调用train_model，第一次False，第二次 True
    if scVI_batch:
        batch_train, batch_test = batch # 批次数据也要分成两部分，训练集和测试集
    
    if step is None: # 默认为True
        step = model.train_step # 在scVIModel中，train_step = optimizer.minimize(self.loss)
        
    batch_size = 128
    iterep = int(expression_train.shape[0]/float(batch_size))-1 # 一共几个批次，这里是 int(2253/128)-1 = 16
    
    training_history = {"t_loss":[], "v_loss":[], "time":[], "epoch":[]}
    training_history["n_hidden"] = model.n_hidden  # 128
    training_history["model"] = model.__class__.__name__
    training_history["n_input"] = model.n_input # 输入节点数，即列数 or 特征数 or 基因数 由数据决定，这里是558
    training_history["dropout_nn"] = model.dropout_rate # 0.1
    training_history["dispersion"] = model.dispersion # "gene"
    training_history["n_layers"] = model.n_layers # 1
    if kl is None: #默认为None
        warmup = lambda x: np.minimum(1, x / 400.)
    else:
        warmup = lambda x: kl
    
    begin = time.time()  # 记录下训练开始前的时间点  
    for t in range(iterep * num_epochs + 1): # num_epochs 第一次是250，第二次是120，所以t的循环范围 第一次是0到 16*250，第二次是16*120
        # warmup
        end_epoch, epoch = t % iterep == 0, t / iterep
        # 当t=16的倍数时， end_epoch=True，共有250或120次True，也就是num_epoch
        # epoch = 0/16, 1/16 ... 120或250
        kl = warmup(epoch)
        # 当 epoch<=400时，kl=warmup(1)=None，两次epoch都<=400
    
        # arange data in batches 在批次中整理数据
        index_train = np.random.choice(np.arange(expression_train.shape[0]), size=batch_size) # 从[0,1,...,2252]中随机抽取128个元素（可重复抽取），排列为一个一位数组
        x_train = expression_train[index_train].astype(np.float32) # 在训练集中，选取对应索引的数据，共128个样本

        # prepare data dictionaries 准备数据字典
        dic_train = {model.expression: x_train, model.training_phase:True, model.kl_scale:kl}
        dic_test = {model.expression: expression_test,  model.training_phase:False, model.kl_scale:kl}
        
        if scVI_batch:  # 第一次 False，第二次 True
            b_train = batch_train[index_train] # batch数据的训练集
            # 再在数据字典中，加入batch和mmd正则化的记录
            dic_train[model.batch_ind] = b_train
            dic_train[model.mmd_scale] = 10
            dic_test[model.batch_ind] = batch_test
            dic_test[model.mmd_scale] = 10

        
        # run an optimization set 进行一次优化
        _, l_tr = sess.run([model.train_step, model.loss], feed_dict=dic_train) # run train_step 则进行一次优化，用l_tr记录loss记录训练集损失函数

        if end_epoch:   # 每当一次end_epoch=True，就表示一次epoch训练结束，一次epoch一共优化了16次
            
            now = time.time() # 记录下每次epoch结束时的时间点

            l_t = sess.run((model.loss), feed_dict=dic_test)    # 用l_t记录开始和每次epoch结束时的测试集损失函数，全部一共num_epoch个值
            
            # 用字典记录训练过程中的数据
            training_history["t_loss"].append(l_tr)
            training_history["v_loss"].append(l_t)
            training_history["time"].append(format_time(int(now-begin)))
            training_history["epoch"].append(epoch)
            
            ## 自己加的，每10个Epoch打印训练Loss和测试Loss
            if epoch % 20 == 0:
                print("Epoch:", epoch, "| Train Loss:", l_tr, "| Test Loss:", l_t)
            
            if np.isnan(l_tr):  # 如果l_tr为NaN，则跳过一次优化
                break
        
    return training_history


# In[2]:
# # parameters

batch_size = 128
learning_rate = 0.0005 # default = 0.0005
epsilon = 0.01
num_epochs = 300


# express data
X = pd.read_csv("../datasets/pbmc68k_filtered_cell.csv", index_col=0)
expression_data = np.array(X, dtype=np.int)
#selected = np.std(expression_data, axis=0).argsort()[-1000:][::-1]
#expression_data = expression_data[:, selected]

clusters = pd.read_csv("../datasets/pbmc68k_filtered_cell_labels.csv", index_col=0)
cell_type = np.array(clusters['cell_type'])


# In[8]:


from sklearn.model_selection import train_test_split
expression_train, expression_test, c_train, c_test = train_test_split(expression_data, cell_type, random_state=0, test_size=0.01)
train_index = pd.DataFrame(expression_train)._stat_axis.values.tolist()
test_index = pd.DataFrame(expression_test)._stat_axis.values.tolist()

# batch = pd.read_csv("../mouse_cell_atlas_labels.csv", index_col=0)
# b_train = np.array(clusters['batch'].values, dtype=np.int)[train_index]
# b_test = np.array(clusters['batch'].values, dtype=np.int)[test_index]



# In[10]:

## # # Computational graph 训练模型

tf.reset_default_graph()
l_mean, l_var = np.mean(np.log(np.sum(expression_train, axis=1))), np.var(np.log(np.sum(expression_train, axis=1)))
expression = tf.placeholder(tf.float32, (None, expression_train.shape[1]), name='x')
#batch_ind = tf.placeholder(tf.int32, [None], name='b_ind')
kl_scalar = tf.placeholder(tf.float32, (), name='kl_scalar')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
training_phase = tf.placeholder(tf.bool, (), name='training_phase')
mmd_scalar = tf.placeholder(tf.float32, [], name='l')

model = scVI.scVIModel(expression= expression, n_latent=10, n_layers=1,
                       batch_ind=None, num_batches=2,
                       kl_scale=kl_scalar,
                       apply_mmd=False, phase=training_phase, mmd_scale=mmd_scalar, optimize_algo=optimizer,
                       library_size_mean=l_mean, library_size_var=l_var, dispersion="gene")

# Session creation
sess = tf.Session() 
sess.run(tf.compat.v1.global_variables_initializer())
result = train_model(model, (expression_train, expression_test), sess, num_epochs=num_epochs)#, batch=(b_train, b_test))


# In[ ]:


plt.plot(result["epoch"], result["t_loss"], label='train loss' )
plt.plot(result["epoch"], result["v_loss"], label='validation loss')
plt.title("Loss", fontsize=16)
plt.xlabel("number of epochs", fontsize=12)
plt.ylabel("objective function",fontsize=12)
plt.legend(fontsize=14)
plt.tight_layout()
# # Evaluation methods

# In[ ]:


def eval_latent(model, data, sess, batch=None):
    dic_full = {model.expression: data, model.training_phase:False, model.kl_scale:1}
    if batch is not None:
        dic_full[model.batch_ind] = batch
        dic_full[model.mmd_scale] = 0 
    return sess.run(model.z, feed_dict=dic_full)

latent = eval_latent(model, expression_train, sess)#, batch=b_train)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.manifold import TSNE


def show_tSNE(latent, labels, return_tSNE=False):
    unique_label = np.unique(labels)
    labels_pred = KMeans(n_clusters=len(unique_label)).fit_predict(latent)
    print("silhouette_score =", silhouette_score(latent, labels))
    print("ARI =", ARI(labels, labels_pred))
    print("NMI =", NMI(labels, labels_pred))
    
    if latent.shape[1] != 2:
        latent = TSNE().fit_transform(latent)
    
    latent_df = pd.DataFrame(latent, columns=['tsne1','tsne2'])
    latent_df['labels'] = labels
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
    


tsne = show_tSNE(latent, c_train, return_tSNE=True)
# =============================================================================
#     ## 批次效应散点图
#     batch = pd.read_csv('../retina_labels.csv',index_col=0)['batch'].values
#     print("batch cluster silhouette_score =", silhouette_score(tsne, batch))
#     plt.figure(figsize=(8, 8))
#     plt.scatter(tsne[:,0],tsne[:,1], c=batch, cmap=plt.get_cmap('brg', 2), edgecolors='none', s=20)
#     plt.axis("off")
#     plt.tight_layout()
# =============================================================================


