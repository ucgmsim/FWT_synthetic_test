# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 18:09:29 2019

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Generate data on commute times.
#size, scale = 1000, 10
#commutes = pd.Series(np.random.gamma(scale, size=size) ** 1.5)
#commutes.plot.hist(grid=True, bins=20, rwidth=0.9,color='#607c8e')
i=1
#commutes = pd.Series(np.loadtxt('rwm_all_iter_'+str(i)+'.txt'))
rwm_all1 = np.loadtxt('rwm_all_iter_'+str(i)+'.txt')
i=8
#commutes = pd.Series(np.loadtxt('rwm_all_iter_'+str(i)+'.txt'))
rwm_all2 = np.loadtxt('rwm_all_iter_'+str(i)+'.txt')
#rwm_all = np.loadtxt('rwm_all.txt')
#r_mean1 = np.mean(rwm_all1)
#r_mean2 = np.mean(rwm_all2)
count=0
for i in range(0,699):
    if (rwm_all1[i]>100):
        rwm_all1[i] = 0
        rwm_all2[i] = 0
        count=count+1
commutes = pd.Series(rwm_all1[0:699])
commutes2 = pd.Series(rwm_all2[0:699])
print(count)
#commutes = pd.Series(np.log10(rwm_all1[0:699]))
#commutes = commutes/np.max(commutes)*10
#commutes2= pd.Series(np.log10(rwm_all2[0:699]))
#commutes.plot.hist(grid=True, bins=[0.5, 0.75, 1, 1.25, 1.5, 1.75,  2, 2.25, 2.5], rwidth=0.9, color='#607c8e',histtype='stepfilled',align='mid')
commutes.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
plt.vlines(np.median(commutes),0, 250, colors='k', linestyles='dashed')
commutes2.plot.hist(grid=True, bins=10, rwidth=0.9, color='r')
plt.vlines(np.median(commutes2),0, 250, colors='r', linestyles='dashed')
plt.title('Relative waveform misfit distribution',fontsize=14)
plt.ylabel('Waveform count',fontsize=14)
#plt.xlabel('RWM (log-scale)',fontsize=14)
plt.xlabel('RWM',fontsize=14)
plt.grid(axis='y', alpha=0.75)
