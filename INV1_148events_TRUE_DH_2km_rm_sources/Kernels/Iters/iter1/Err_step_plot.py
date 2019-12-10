#!/usr/bin/env python2i
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 10:50:28 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt

iN=5
Err=np.zeros((iN,1))
Step=np.zeros((iN,1))
for i in range(1,iN+1):
    err_iter='Dump/err_iter_'+str(i)+'.dat'
    step_iter='Dump/step_iter_'+str(i)+'.dat'
#    err_iter='Dump/Dump_04_05_19_2iters/err_iter_'+str(i)+'.dat'
    f_err=open(err_iter,'r')
    Err1=np.fromfile(f_err,dtype=np.float64)
    print(Err1)	
    Err[i-1]=Err1
    
    f_step=open(step_iter,'r')
    Step1=np.fromfile(f_step,dtype=np.float64)    
    print(Step1)	
    Step[i-1]=Step1

Err=Err/np.max(Err)

plt.figure(figsize=(10,2.5))
plt.subplot(1,2,1)
#plt.figure(figsize=(10,1.25))    
plt.plot(Err,c='r')
plt.ylabel('RWM reduction')
plt.xlabel('Iteration')
plt.xlim([0,iN-1])
#plt.ylim([350,np.max(Err)])
xint=range(0,iN-1)
plt.xticks(xint)
#plt.savefig('err.png',dpi=200)
#plt.show()
plt.subplot(1,2,2)
#plt.figure(figsize=(10,1.25))    
plt.plot(Step,c='r')
plt.ylabel('Optimal steps')
plt.xlabel('Iteration')
plt.xlim([0,iN-1])
plt.ylim([0,0.5])
xint=range(0,iN-1)
plt.xticks(xint)
#plt.savefig('err.png',dpi=200)
plt.show()
