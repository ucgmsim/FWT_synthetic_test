#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
import numpy as np
import math
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

def precondition_matrix(nx,ny,nz,R,R_nf):
    Tp=np.ones((ny,nz,nx))
    nR=R.shape[0]
    for i in range(1,nx+1):
        for j in range(1,ny+1):
            for k in range(1,nz+1):                                               
                for ir in range(1,nR+1):
                    dr=((i-R[ir-1,0])**2+(j-R[ir-1,1])**2+(k-R[ir-1,2])**2)**(0.5)
                    if (dr<R_nf):
                        Tp[j-1,k-1,i-1]*=np.power(math.sin(math.pi/2*dr/R_nf),3)
                        
    return Tp
   
# Read_Dev_strain_ex2.m
nx=65
ny=65
nz=40

nxyz=nx*ny*nz

#dx=0.4
#dy=0.4
#dz=0.4
#dt=0.02

#source_file='../../../StatInfo/SOURCE.txt'
station_file='../../../StatInfo/STATION.txt'

#nShot, S = read_source(source_file)
nRec, R = read_source(station_file)                       

##########################################################

#R_nf=10
R_nf=5
#R_nf=30
print('Tape matrix all receivers ')     
Tp=precondition_matrix(nx,ny,nz,R,R_nf)
Tp_arr=Tp.reshape(-1)
tape_file='Dump/Tp_rec.txt'
np.savetxt(tape_file, Tp_arr)

print('finish creating tape matrices')
