# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 23:30:03 2021

@author: Administrator
"""

import pandas as pd
import numpy as np


data = pd.read_csv('twogroupsimulation.csv', sep=',', index_col=0).values
# data = pd.read_csv('sixgroupsimulation.csv', sep=',', index_col=0).values

indices = np.random.choice(np.arange(data.size), replace=False,
                           size=int(data.size * 0.5))

data = data.reshape(data.size, )

data[indices] = 0

data = data.reshape(2000, 197)
# data = data.reshape(2000, 199)

pd.DataFrame(data).to_csv("twogroupsimulation_0.50dropout.csv")
# pd.DataFrame(data).to_csv("sixgroupsimulation_0.50dropout.csv")