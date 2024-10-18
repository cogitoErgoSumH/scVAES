# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 00:18:55 2021

@author: Administrator
"""


## 基因表达热力图
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

idx_c = random.sample(range(0, 2000), 20)
idx_g = random.sample(range(0, 199), 20)

orig = pd.read_csv("../datasets/sixgroupsimulation.csv", index_col=0).values#[idx_c,:][:, idx_g]
drop = pd.read_csv("../datasets/sixgroupsimulation_0.50dropout.csv", index_col=0).values#[idx_c,:][:, idx_g]
stu = pd.read_csv("./x_hat_six_050_student_final.csv", index_col=0).values#[idx_c,:][:, idx_g]
gau = pd.read_csv("./x_hat_six_050_gaussian_final.csv", index_col=0).values#[idx_c,:][:, idx_g]
nb = pd.read_csv("./x_hat_six_050_nb_final.csv", index_col=0).values#[idx_c,:][:, idx_g]
zinb = pd.read_csv("./x_hat_six_050_zinb_final.csv", index_col=0).values#[idx_c,:][:, idx_g]

orig = np.log(orig + 1)
drop = np.log(drop + 1)

# mean absolute error
stu_mae = np.abs(stu-orig).sum() / (2000*199)
gau_mae = np.abs(gau-orig).sum() / (2000*199)
nb_mae = np.abs(nb-orig).sum() / (2000*199)
zinb_mae = np.abs(zinb-orig).sum() / (2000*199)

print('stu_mae =', stu_mae)
print('gau_mae =', gau_mae)
print('nb_mae =', nb_mae)
print('zinb_mae =', zinb_mae)

orig_20 = orig[idx_c,:][:, idx_g]
drop_20 = drop[idx_c,:][:, idx_g]
stu_20 = stu[idx_c,:][:, idx_g]
gau_20 = gau[idx_c,:][:, idx_g]
nb_20 = nb[idx_c,:][:, idx_g]
zinb_20 = zinb[idx_c,:][:, idx_g]

# =============================================================================
# np.savetxt("six_orig_20.csv", orig_20, delimiter=',')
# np.savetxt("six_drop_20.csv", drop_20, delimiter=',')
# np.savetxt("six_stu_20.csv", stu_20, delimiter=',')
# np.savetxt("six_gau_20.csv", gau_20, delimiter=',')
# np.savetxt("six_nb_20.csv", nb_20, delimiter=',')
# np.savetxt("six_zinb_20.csv", zinb_20, delimiter=',')
# =============================================================================

orig_20 = (orig_20 -np.min(orig_20)) / (np.max(orig_20) - np.min(orig_20))
drop_20 = (drop_20 -np.min(drop_20)) / (np.max(drop_20) - np.min(drop_20))
stu_20 = (stu_20 -np.min(stu_20)) / (np.max(stu_20) - np.min(stu_20))
gau_20 = (gau_20 -np.min(gau_20)) / (np.max(gau_20) - np.min(gau_20))
nb_20 = (nb_20 -np.min(nb_20)) / (np.max(nb_20) - np.min(nb_20))
zinb_20 = (zinb_20 -np.min(zinb_20)) / (np.max(zinb_20) - np.min(zinb_20))

fig1, axes1 = plt.subplots(nrows=2,ncols=3, figsize=(18,10))
sns.heatmap(orig_20, cmap='YlGnBu', ax=axes1[0][0])
sns.heatmap(drop_20, cmap='YlGnBu', ax=axes1[0][1])
sns.heatmap(stu_20, cmap='YlGnBu', ax=axes1[0][2])
sns.heatmap(gau_20, cmap='YlGnBu', ax=axes1[1][0])
sns.heatmap(nb_20, cmap='YlGnBu', ax=axes1[1][1])
sns.heatmap(zinb_20, cmap='YlGnBu', ax=axes1[1][2])

fig2, axes2 = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
sns.heatmap(np.abs(stu_20-orig_20), vmin=0, vmax=1, cmap='Blues', ax=axes2[0][0])
sns.heatmap(np.abs(gau_20-orig_20), vmin=0, vmax=1, cmap='Blues', ax=axes2[0][1])
sns.heatmap(np.abs(nb_20-orig_20), vmin=0, vmax=1, cmap='Blues', ax=axes2[1][0])
sns.heatmap(np.abs(zinb_20-orig_20), vmin=0, vmax=1, cmap='Blues', ax=axes2[1][1])
