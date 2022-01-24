# -*- coding: utf-8 -*-
"""
Created on Tue May  4 17:34:07 2021

@author: Yaren
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

dataset = pd.read_csv('Credit_Card_Applications.csv')
dataset.info()

x = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values

scaler = MinMaxScaler(feature_range = (0,1))
x = scaler.fit_transform(x)

som = MiniSom(x=10, y=10, input_len= 15, sigma= 1.0, learning_rate = 0.5)
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 100)

bone()
pcolor( som.distance_map().T, cmap = "Blues")
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, k in enumerate(x):
    w = som.winner(k)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 1)
show()

map_ = som.win_map(x)
frauds = np.concatenate((map_[(1,2)], map_[(1,8)], map_[(2,3)], map_[(3,3)]), axis = 0)
frauds = scaler.inverse_transform(frauds) 

print('Fraud Customer IDs')
for i in frauds[:, 0]:
  print(int(i))