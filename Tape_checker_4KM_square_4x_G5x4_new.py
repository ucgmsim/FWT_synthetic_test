#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
import numpy as np
import math
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
#import pdb; 
def Vsp_read(nx,ny,nz,snap_file):
    fid=open(snap_file,'r')
    sek=np.fromfile(fid, dtype=np.float32)
    Vp=np.reshape(sek,[ny,nz,nx])
    return Vp  

def write_model_Vsp_name(rho,Vs,Vp,fname):
    [ny,nz,nx]=Vs.shape
    #[nx,nz,ny]=Vs.shape
    
    model_file='rho3d'+fname+'.d'
    fid=open(model_file,'wb')
    sek1 = np.array(rho, dtype=np.float32)
    sek1.astype('float32').tofile(fid)
    
    model_file0='vs3d'+fname+'.s'
    model_file1='vp3d'+fname+'.p'

    fid00=open(model_file0,'wb')
    sek2 = np.array(Vs, dtype=np.float32)
    sek2.astype('float32').tofile(fid00)

    fid01=open(model_file1,'wb')
    sek3 = np.array(Vp, dtype=np.float32)
    sek3.astype('float32').tofile(fid01)
    
def precondition_matrix(nx,ny,nz,R,R_nf):
    Tp=np.ones((ny,nz,nx))
    for i in range(0,nx,20):
        for j in range(0,ny,20):
            
            if((int(i/20.0)+int(j/20.0))%2==0):
                Tp[j:j+10,:,i:i+10]=0.8
            if((int(i/20.0)+int(j/20.0))%2==1):
                Tp[j:j+10,:,i:i+10]=1.2
                
#    Tp = ndimage.gaussian_filter(Tp, 5)
    Tp = ndimage.gaussian_filter(Tp, 2)                                      
                       
    return Tp
   
# Read_Dev_strain_ex2.m
nx=88
ny=88
nz=60
dx=4; dy=4;

nxyz=nx*ny*nz
           
rx=np.arange(5,nx,20)
ry=np.arange(5,ny,20)           

nrx=len(rx)
nry=len(ry)

Nr=nrx*nry

R=np.zeros((Nr,3))
for i in np.arange(0,nrx):
    for j in np.arange(0,nry):
        R[j*nrx+i,0]=rx[i]
        R[j*nrx+i,1]=ry[j]
        R[j*nrx+i,2]=1                     

##########################################################

R_nf=8
#R_nf=5
##R_nf=20
#print('Tape matrix all receivers ')     
Tp=precondition_matrix(nx,ny,nz,R,R_nf)
Tp_arr=Tp.reshape(-1)
tape_file='Tp_checker_square_4x_G5x4_new.txt'
np.savetxt(tape_file, Tp_arr)
#
print('finish creating tape matrices')

plt.figure(figsize=(10,10))
   
for j in range(0,Nr):
    plt.plot((R[j,0])*dx,(R[j,1])*dx,'ob')  

plt.title('Test configuration', loc='center') 
plt.xlim([0,nx*dx])
plt.ylim([0,ny*dy])
plt.grid()
plt.xlabel('x-axis (km)')
plt.ylabel('y-axis (km)')


K=Tp[:,20,:]
fig = plt.figure(figsize=(10,10))
plt.imshow(K)
plt.show()

#snap_file1='/home/andrei/workspace/GMPlots/NorthCanVM_Events/NewVM_20191219/3366146/vs3dfile.s'
#snap_file2='/home/andrei/workspace/GMPlots/NorthCanVM_Events/NewVM_20191219/3366146/vp3dfile.p'
#snap_file3='/home/andrei/workspace/GMPlots/NorthCanVM_Events/NewVM_20191219/3366146/rho3dfile.d'
snap_file1='vs3dfile_smooth_4km.s'
snap_file2='vp3dfile_smooth_4km.p'
snap_file3='rho3dfile_smooth_4km.d'

Vs1=Vsp_read(nx,ny,nz,snap_file1)
Vp1=Vsp_read(nx,ny,nz,snap_file2)
rho1=Vsp_read(nx,ny,nz,snap_file3)

K=Vs1[:,20,:]
fig = plt.figure(figsize=(10,10))
plt.imshow(K)
plt.show()


fname_true='file_true_square_4x_G5x4_new'
#fname_true='file_true_square_2x_G2x4'
Vs=np.multiply(Vs1,Tp)
Vp=np.multiply(Vp1,Tp)
rho=rho1

write_model_Vsp_name(rho,Vs,Vp,fname_true)

K=Vs1[35,:,:]
fig = plt.figure(figsize=(10,10))
plt.imshow(K)
plt.show()
