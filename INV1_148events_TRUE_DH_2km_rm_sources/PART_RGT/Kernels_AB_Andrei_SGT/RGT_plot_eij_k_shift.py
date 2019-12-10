#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
import numpy as np
import matplotlib.pyplot as plt
#import pdb; 
from scipy.signal import butter, lfilter
from scipy import signal

def conv_wf(eii3,eiib,nt,dt):
    """
         Convolve forward and backward
    """
    (nt,ny,nz,nx) = eii3.shape
#    nt=nv+1
    sum_xx= np.zeros((ny,nz,nx))
    for it in range(1,nt):
#        sum_xx=sum_xx+dt*np.multiply(eii3[:,:,:,it-1],eiib[:,:,:,nt-it])
        sum_xx=sum_xx+dt*np.multiply(eii3[it-1,:,:,:],eiib[nt-it,:,:,:])     
    return sum_xx

def Vsp_read(nx,ny,nz,snap_file):
    fid=open(snap_file,'r')
    sek=np.fromfile(fid, dtype='<f4')
    #print(sek.shape)
    Vp=np.reshape(sek,[ny,nz,nx])
#    Vp=np.reshape(sek,[nx,nz,ny])
#    Vp=np.zeros((nx,nz,ny))
#    for k in range(1,nx):
#        for j in range(1,nz):
#            for i in range(1,ny):                
#                Vp[k-1,j-1,i-1]=Vp1[i-1,j-1,k-1]
    return Vp
    
def time_shift_emod3d(data,delay_Time,dt):
    n_pts = len(data)
    ndelay_Time = int(delay_Time/(dt))
    data_shift = np.zeros(data.shape)
    data_shift[0:n_pts-ndelay_Time] = data[ndelay_Time:n_pts]
    return data_shift

# Read_Dev_strain_ex2.m
nx=65
ny=65
#nz=75
nz=40
nts=2000
nt=400
nxyz=nx*ny*nz

num_pts=2000; dt=0.08;
t = np.arange(num_pts)*dt

fs = 1/dt
lowcut = 0.05
#highcut = 0.4
highcut = 0.1

fc = highcut  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(4, w, 'low')

flo=0.1
delay_Time=(3/flo)+10

statname='REHS'
with open('Dev_Strain/'+statname+'/RGT_X/fwd01_xyzts.exx', 'rb') as fid:
    data_array = np.fromfile(fid, np.float32)
    data_array=data_array[0:14]
    dx=data_array[8]
    dy=data_array[9]
    dz=data_array[10]
    dt=data_array[11]
    del data_array
    
snap_file1='../../Model/Models/vs3dfile_h.s';
snap_file2='../../Model/Models/vp3dfile_h.p';
#snap_file1='vs3dfile.s';
#snap_file2='vp3dfile.p';
snap_file3='../../Model/Models/rho3dfile.d';
Vs=Vsp_read(nx,ny,nz,snap_file1)
Vp=Vsp_read(nx,ny,nz,snap_file2)
rho=Vsp_read(nx,ny,nz,snap_file3)

#H=Vs[150,:,:]
#fig = plt.figure(figsize=(6, 4))
#plt.imshow(H)
#plt.show()

Lamda=np.multiply(rho,(np.multiply(Vp,Vp)-2*np.multiply(Vs,Vs)));
Mu=np.multiply(rho,np.multiply(Vs,Vs))
Kappa=Lamda+2/3*Mu

#mainfolder_fig='/home/user/workspace/GMPlots/Figs/'
#eijp_names = {'eii3','exxp','eyyp','exyp','exzp','eyzp'};
#statname='CBGS'
eijp_names = {'exx'}
fx_file='Dev_Strain/'+statname+'/RGT_X/fwd01_xyzts.';
fy_file='Dev_Strain/'+statname+'/RGT_Y/fwd01_xyzts.';
fz_file='Dev_Strain/'+statname+'/RGT_Z/fwd01_xyzts.';
#bw_file='Dev_Strain/fwd01_xyzts.';
for eij_name in eijp_names:
    matrix_file_fx=fx_file+eij_name
    matrix_file_fy=fy_file+eij_name
    matrix_file_fz=fz_file+eij_name

    fid1=open(matrix_file_fx, 'rb')
    matrix_dummy_fw=np.fromfile(fid1, np.float32)
    matrix_dummy_fx=matrix_dummy_fw[15:]

    fid1=open(matrix_file_fy, 'rb')
    matrix_dummy_fw=np.fromfile(fid1, np.float32)
    matrix_dummy_fy=matrix_dummy_fw[15:]

    fid1=open(matrix_file_fz, 'rb')
    matrix_dummy_fw=np.fromfile(fid1, np.float32)
    matrix_dummy_fz=matrix_dummy_fw[15:]
    
    
    if eij_name=='exx':
#        eii3 = np.reshape(matrix_dummy_fw,[nx,nz,ny,nt]) 
#        eiib = np.reshape(matrix_dummy_bw,[nx,nz,ny,nt])
        eiix = np.reshape(matrix_dummy_fx,[nt,ny,nz,nx]) 
        eiiy = np.reshape(matrix_dummy_fy,[nt,ny,nz,nx])        
        eiiz = np.reshape(matrix_dummy_fz,[nt,ny,nz,nx]) 
#        sum1=9*conv_wf(eii3,eiib,nt,dt)
#        GM=-2*np.multiply(sum1,Mu)   
        eii3_fx=eiix[:,30,30,20]
        eii3_fy=eiiy[:,30,30,20]
        eii3_fz=eiiz[:,30,30,20]
        eii3_fx = time_shift_emod3d(eii3_fx,delay_Time,dt*5)
        eii3_fy = time_shift_emod3d(eii3_fy,delay_Time,dt*5)
        eii3_fz = time_shift_emod3d(eii3_fz,delay_Time,dt*5)
#        print('e3/eb='+str(np.max(np.abs(eii3_fw))/np.max(np.abs(eii3_bw))))
        plt.figure(figsize=(10,1.25))
        plt.plot(eii3_fx/np.max(np.abs(eii3_fx)),c='k')
        plt.plot(eii3_fy/np.max(np.abs(eii3_fy)),c='b')
        plt.plot(eii3_fz/np.max(np.abs(eii3_fz)),c='r')
        

        plt.gca().legend(('RGT -x','RGT -y','RGT -z'))
        plt.show()
#        del eii3 
#        del eiib

##GM = np.arange(nxyz).reshape((nx,nz,ny))
#GM_arr=GM.reshape(-1)
###GM_arr=GM.ravel()
##
#np.savetxt('GM.txt', GM_arr)

#K=GM[50,:,:]
#fig = plt.figure(figsize=(6, 4))
#plt.imshow(K)
#plt.show()        
#file=open('GM2.txt','w')
#file.write(GM_arr)
#file.close()
        
#    if eij_name=='exxp':
#        exxp = np.reshape(matrix_dummy_fw,[nx,nz,ny,nt]) 
#        exxb = np.reshape(matrix_dummy_bw,[nx,nz,ny,nt])
    
    



#eii3_s=mafs2[ix,iz,iy,:]
#
#plt.figure(figsize=(10,1.25))
#plt.plot(eii3_s,c='k')
#plt.ylabel('eii3_s')
#plt.xlim([0,nt])
#ax = plt.gca()
#ax.axis('off')
