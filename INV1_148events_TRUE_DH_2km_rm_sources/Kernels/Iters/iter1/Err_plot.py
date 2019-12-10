#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 10:50:28 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt

iN=3
Err=np.zeros((iN,1))
for i in range(1,iN+1):
    #err_iter='Dump/Dump_24_07_19_2iters_good_hl_0.05_01Hz_smooth_init_148s/err_iter_'+str(i)+'.dat'
    err_iter='Dump/err_iter_'+str(i)+'.dat'
    f_err=open(err_iter,'r')
    #Err1=np.fromfile(f_err,dtype='<f4')
    Err1=np.fromfile(f_err,dtype=np.float64)

    print(Err1)	
    Err[i-1]=Err1

#Err[2]=26.83039782
plt.figure(figsize=(10,1.25))    
plt.plot(Err/Err[0],c='r')
plt.xlabel('Iteration number')
plt.ylabel('Least squared error')
plt.xlim([0,iN-1])
plt.ylim([np.min(Err)/Err[0],np.max(Err)/Err[0]])
xint=range(1,iN)
plt.xticks(xint)
plt.savefig('err.png',dpi=200)
plt.show()
