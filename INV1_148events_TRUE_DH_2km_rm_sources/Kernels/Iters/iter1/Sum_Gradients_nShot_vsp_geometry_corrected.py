#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
import numpy as np
#import math
import scipy.ndimage as ndimage
#import matplotlib.pyplot as plt
#import pdb; 
def read_source(source_file):
    
    with open(source_file, 'r') as f:
        lines = f.readlines()    
    line0=lines[0].split()
    nShot=int(line0[0])
    S=np.zeros((nShot,3))
    for i in range(1,nShot+1):
        line_i=lines[i].split()
        S[i-1,0]=int(line_i[0])
        S[i-1,1]=int(line_i[1])
        S[i-1,2]=int(line_i[2])
    
    return nShot, S

def read_stat_name(station_file):

    with open(station_file, 'r') as f:
        lines = f.readlines()
    line0=lines[0].split()
    nRec=int(line0[0])
    R=np.zeros((nRec,3))
    statnames = [] 
    for i in range(1,nRec+1):
        line_i=lines[i].split()
        R[i-1,0]=int(line_i[0])
        R[i-1,1]=int(line_i[1])
        R[i-1,2]=int(line_i[2])
        statnames.append(line_i[3])
    return nRec, R, statnames    

nx=65
ny=65
nz=40
nts=2000
nt=500
nxyz=nx*ny*nz

#dx=0.4
#dx=1.0
#dy=dx
#dz=dx
#dt=0.02

source_file='../../../StatInfo/SOURCE.txt'
station_file='../../../StatInfo/STATION.txt'

nShot, S = read_source(source_file)
nRec, R, statnames = read_stat_name(station_file)                    

wr = np.loadtxt('Dump/geo_correlation.txt')
##########################################################
wr_ishot = np.zeros(nShot)
for ishot in range(1,nShot+1):  

    R_ishot_arr=np.loadtxt('Dump/R_ishot_'+str(ishot)+'.txt')
    for i,statname in enumerate(statnames):
        distance=((R[i,1]-S[ishot-1,1])**2+(R[i,2]-S[ishot-1,2])**2+(R[i,0]-S[ishot-1,0])**2)**(0.5)
        
        if ((distance<200) and (distance>0) and (R_ishot_arr[i]==1)):
            wr_ishot[ishot-1] = wr_ishot[ishot-1]+wr[nRec*(ishot-1)+i]
            


Gra_S=np.zeros((ny,nz,nx))
Gra_P=np.zeros((ny,nz,nx))

#for ii in range(75,148+1):
#for ii in range(75,111+1):
for ii in range(1,nShot+1):
    if (S[ii-1,2]<20):    
        print('Sum Gradient Shot '+str(ii))  
        GS_file='All_shots/GS_shot'+str(ii)+'.txt'    
        GP_file='All_shots/GP_shot'+str(ii)+'.txt'
        
        GS_arr=np.loadtxt(GS_file)
        GS=np.reshape(GS_arr,[ny,nz,nx])
        GP_arr=np.loadtxt(GP_file)
        GP=np.reshape(GP_arr,[ny,nz,nx])
        
        #Tp=precondition_matrix(nx,ny,nz,S[ii-1,:],R,R_nf)
        tape_file='Dump/Tp_shot_'+str(ii)+'.txt'
        Tp_arr=np.loadtxt(tape_file)
        Tp=np.reshape(Tp_arr,[ny,nz,nx])
        #Tp=np.ones((ny,nz,nx)) 
        
        Gra_S=Gra_S+np.multiply(GS,Tp)*wr_ishot[ii-1]
        Gra_P=Gra_P+np.multiply(GP,Tp)*wr_ishot[ii-1]
    else:
        print('No kernel calculated for source '+str(ii))        
#Tape boundary
tape_file='Dump/Tp_xyz.txt'
Tp_arr=np.loadtxt(tape_file)
Tp=np.reshape(Tp_arr,[ny,nz,nx]) 

Gra_S=np.multiply(Gra_S,Tp)
Gra_P=np.multiply(Gra_P,Tp)
#Tape receiver
tape_file='Dump/Tp_rec.txt'
Tp_arr=np.loadtxt(tape_file)
Tp=np.reshape(Tp_arr,[ny,nz,nx])

Gra_S=np.multiply(Gra_S,Tp)
Gra_P=np.multiply(Gra_P,Tp)

Gra_S[:,0:2,:]=0
Gra_P[:,0:2,:]=0

Gra_S=ndimage.gaussian_filter(Gra_S, 3)
Gra_P=ndimage.gaussian_filter(Gra_P, 3)

Gra_S_arr=Gra_S.reshape(-1)
np.savetxt('Gra_S.txt', Gra_S_arr)
    
Gra_P_arr=Gra_P.reshape(-1)
np.savetxt('Gra_P.txt', Gra_P_arr)
print('finish Summing Gradients')

print('max Gs='+str(np.max(Gra_S_arr)))
print('max Gp='+str(np.max(Gra_P_arr)))

