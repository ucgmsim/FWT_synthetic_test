#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:52:35 2019

@author: user
"""
import numpy as np
#import os
#import time
#from numpy.linalg import solve
#import Step_length_new
###########################################################################
# Read_Dev_strain_ex2.m
nx=200
ny=200
nz=100
nts=5000
nt=500

dx=1.0
dy=dx
dz=dx

iters=10
#nShot=16
##############################################################################
fi1=open('iNumber.dat','r')
it=int(np.fromfile(fi1,dtype='int64'))
fi1.close()
print('iNumber='+str(it))

################################  
Gra_S_arr=np.loadtxt('Gra_S.txt')
#    Gra_S_arr=Gra_S_arr-np.mean(Gra_S_arr)
Gra_Vs=np.reshape(Gra_S_arr,[ny,nz,nx])

Gra_P_arr=np.loadtxt('Gra_P.txt')
w=it/iters
Gra_P_arr=w*Gra_P_arr+(1-w)*Gra_S_arr
#Gra_P_arr=Gra_P_arr-np.mean(Gra_P_arr)    
Gra_Vp=np.reshape(Gra_P_arr,[ny,nz,nx])
  
gra=np.zeros(2*nx*ny*nz)
grad_last=np.zeros(2*nx*ny*nz)

gra[0:nx*ny*nz] = Gra_S_arr
gra[nx*ny*nz:2*nx*ny*nz] = Gra_P_arr


#gra1=gra
if (it==1):
    gra1=gra
#    
if (it>1):#CG Polak-Ribiere
    grads_last_arr=np.loadtxt('Dump/Grads_iter_'+str(it-1)+'.txt')
    gradp_last_arr=np.loadtxt('Dump/Gradp_iter_'+str(it-1)+'.txt')  
#        
    grad_last[0:nx*ny*nz] = grads_last_arr
    grad_last[nx*ny*nz:2*nx*ny*nz] = gradp_last_arr
#
    beta_N=np.dot(gra,gra-grad_last)
    beta_D=np.dot(gra,grad_last)
    print('beta_N='+str(beta_N)+'and beta_D='+str(beta_D))
    gra1=gra+beta_N/beta_D*grad_last        
        
#################################################################
Gra_Vs_arr=gra1[0:nx*ny*nz] 
max1=1.0*np.max(np.abs(Gra_Vs_arr))
for i in range(1,len(Gra_Vs_arr)+1):
    if Gra_Vs_arr[i-1]>max1:
        Gra_Vs_arr[i-1]=max1
    if Gra_Vs_arr[i-1]<-max1:
        Gra_Vs_arr[i-1]=-max1
Gra_Vs=np.reshape(Gra_Vs_arr,[ny,nz,nx])

Gra_Vp_arr=gra1[nx*ny*nz:2*nx*ny*nz]
max2=1.0*np.max(np.abs(Gra_Vp_arr))
for i in range(1,len(Gra_Vp_arr)+1):
    if Gra_Vp_arr[i-1]>max2:
        Gra_Vp_arr[i-1]=max2
    if Gra_Vp_arr[i-1]<-max2:
        Gra_Vp_arr[i-1]=-max2
Gra_Vp=np.reshape(Gra_Vp_arr,[ny,nz,nx])

print('write gradient iter'+str(it))
np.savetxt('Dump/Grads_iter_'+str(it)+'.txt', Gra_Vs_arr)   
np.savetxt('Dump/Gradp_iter_'+str(it)+'.txt', Gra_Vp_arr)  
##############################