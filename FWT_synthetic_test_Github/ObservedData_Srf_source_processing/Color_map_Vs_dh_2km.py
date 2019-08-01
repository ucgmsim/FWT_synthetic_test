#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:00:02 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
#import os

def read_LL(Vm_LL_file):

    with open(Vm_LL_file, 'r') as f:
        lines = f.readlines()
    nxy=65*65
    S=np.zeros((nxy,2))
    for i in range(0,nxy):
        line_i=lines[i].split()
        S[i,0]=float(line_i[0])
        S[i,1]=float(line_i[1])

    return nxy, S
    
def Vsp_read(nx,ny,nz,snap_file):
    fid=open(snap_file,'r')
    sek=np.fromfile(fid, dtype='<f4')
    Vp=np.reshape(sek,[ny,nz,nx])
    return Vp    

def write_Vs_color_map(S,Vs_h,depth,Vs_color_file):
    
    nxy=len(S)
    print(Vs_color_file)
    fid = open(Vs_color_file,'w')
    fid.write("%s\n" %('CanVMs'))
    fid.write("%s\n" %'units of g')
    fid.write("%s\n" %'polar:invert, t-30 1k:g-surface,nns-12m,landmask,contours')    
    fid.write("%s\n" %'2.5 4.0 0.25 0.5')   
    fid.write("%s\n" %'1 black')       
    fid.write("%s\n" %('Vs map at '+str(depth)+'km'))    
    count=0
    while count<nxy:
        fid.write("%20f%20f%10f \n" %(S[count,0],S[count,1],Vs_h[count]))        
        count=count+1

#nz=75
nz=40
nx=65
ny=65
dx=2.0
dy=dx
dz=dx

x=dx*np.arange(1,nx+1,1)-dx
y=dy*np.arange(1,ny+1,1)-dy
z=dz*np.arange(1,nz+1,1)-dz

#snap_file1='vs3dfile_h.s'
#snap_file2='vp3dfile_h.p'
#snap_file3='rho3dfile_h.d'

snap_file1='vs3dfile_iter_5.s'
snap_file2='vp3dfile_iter_5.p'
snap_file3='rho3dfile_iter_5.d'

Vs1=Vsp_read(nx,ny,nz,snap_file1)
Vp1=Vsp_read(nx,ny,nz,snap_file2)
rho1=Vsp_read(nx,ny,nz,snap_file3)
#Depth [km] to attract the plane view
depth=1 
nz0=int(depth/dx)
Vs_h=  Vs1[:,nz0,:]
Vs_h= Vs_h.reshape(-1)

Vm_LL_file='model_coords_rt01-h2.000'
nxy, S = read_LL(Vm_LL_file)

#Vs_color_file='Vs_h.txt'
Vs_color_file='Vs_inv.txt'
write_Vs_color_map(S,Vs_h,depth,Vs_color_file)