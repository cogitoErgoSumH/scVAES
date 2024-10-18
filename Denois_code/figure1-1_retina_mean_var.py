# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 23:30:03 2021

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('retina.csv', sep=',', index_col=0).values

mean = np.mean(data, axis=0)
var = np.var(data, axis=0)

plt.figure(figsize=(8,6))
plt.scatter(mean, var)
plt.plot([0,np.max(mean)],[0,np.max(mean)])
plt.xlabel('Mean', fontsize=14)
plt.ylabel('Var', fontsize=14)
plt.xlim((0,10))
plt.ylim((0,30))
plt.tick_params(labelsize=14)

