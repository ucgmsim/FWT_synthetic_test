#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
import numpy as np
#import matplotlib.pyplot as plt
#import pdb; 
#import multiprocessing as mp
#from multiprocessing import Pool

def conv_eij_source(e_ij_source, source):
   
#    x = np.arange(0,nt,1)
#    pool = mp.Pool(processes=4)
#    arr_dt_sumxy=pool.map(conv_wf_it_2,x)
#    pool.close()
#    pool.join()   
    nt = len(source)
    sum_ij=e_ij_source(0)
    
    for it in np.arange(0,nt,1):    
        sum_ij=sum_ij+e_ij_source(it)*dt*source(it)           
#    print('done'+eij_name)    
        
    return sum_ij
   
def read_strain_source_location(eij_name,nt0,nx,ny,nz,dz,Sloc):
    
    bw_file='../../Dev_Strain/adj01_xyzts.'
    
    matrix_file_fw=bw_file+eij_name
    
    fid1=open(matrix_file_fw, 'rb')
    matrix_dummy_fw=np.fromfile(fid1, np.float32)
    matrix_dummy_fw=matrix_dummy_fw[15:]
    e_ij_all = matrix_dummy_fw.reshape([nt0,ny,nz,nx])
    e_ij_source = e_ij_all[:,int(Sloc[1])-1,int(Sloc[2])-1,int(Sloc[0])-1]
    e_ij_source_z_der = (e_ij_all[:,int(Sloc[1])-1,int(Sloc[2]),int(Sloc[0])-1] - e_ij_all[:,int(Sloc[1])-1,int(Sloc[2])-2,int(Sloc[0])-1])/(2*dz)
    
       
    return e_ij_source, e_ij_source_z_der

def Vsp_read(nx,ny,nz,snap_file):
    fid=open(snap_file,'r')
    sek=np.fromfile(fid, dtype='<f4')
    print(sek.shape)
    Vp=np.reshape(sek,[ny,nz,nx])
    return Vp

def read_source_new(source_file):
    
    with open(source_file, 'r') as f:
        lines = f.readlines()    
    line0=lines[0].split()
    nShot=int(line0[0])
    S=np.zeros((nShot,3))
    sNames=[]
    for i in range(1,nShot+1):
        line_i=lines[i].split()
        S[i-1,0]=line_i[0]
        S[i-1,1]=line_i[1]
        S[i-1,2]=line_i[2]
        sNames.append(line_i[3])
    
    return nShot, S, sNames    

# Read_Dev_strain_ex2.m
nx=65
ny=65
#nz=75
nz=40
nts=2000
nt0=400
nxyz=nx*ny*nz

source_file = '../../StatInfo/SOURCE.txt'
nShot, S, sNames = read_source_new(source_file)

Sloc =S[0,:]
source = np.zeros([nt0,1])

M0=np.zeros([3,3])
Mxx=M0[0,0];Myy=M0[1,1];Mzz=M0[2,2];
Mxy=M0[0,1];Mxz=M0[0,2];Myz=M0[1,2];

with open('../../Dev_Strain/fwd01_xyzts.exx', 'rb') as fid:
    data_array = np.fromfile(fid, np.float32)
    data_array=data_array[0:14]
    dx=data_array[8]
    dy=data_array[9]
    dz=data_array[10]
    dt=data_array[11]
    del data_array
    
#mainfolder_fig='/home/user/workspace/GMPlots/Figs/'
eijp_names = {'exx','eyy','ezz','exy','exz','eyz'};
#bw_file='../../Dev_Strain/adj01_xyzts.';
CMT_z_der = 0

for eij_name in eijp_names:
    print(eij_name)
    if eij_name=='exx':
        e_xx_source, e_xx_source_z_der = read_strain_source_location(eij_name,nt0,nx,ny,nz,dz,Sloc)
        Mxx_der = conv_eij_source(e_xx_source, source)
        CMT_z_der = CMT_z_der+conv_eij_source(e_xx_source_z_der, source)*Mxx
        
    if eij_name=='eyy':
        e_yy_source, e_yy_source_z_der = read_strain_source_location(eij_name,nt0,nx,ny,nz,dz,Sloc)
        Myy_der = conv_eij_source(e_yy_source, source)
        CMT_z_der = CMT_z_der+conv_eij_source(e_yy_source_z_der, source)*Myy

    if eij_name=='ezz':
        e_zz_source, e_zz_source_z_der = read_strain_source_location(eij_name,nt0,nx,ny,nz,dz,Sloc)
        Mzz_der = conv_eij_source(e_zz_source, source)
        CMT_z_der = CMT_z_der+conv_eij_source(e_zz_source_z_der, source)*Mzz
        
    if eij_name=='exy':
        e_xy_source, e_xy_source_z_der = read_strain_source_location(eij_name,nt0,nx,ny,nz,dz,Sloc)
        Mxy_der = 2*conv_eij_source(e_xy_source, source)
        CMT_z_der = CMT_z_der+2*conv_eij_source(e_xy_source_z_der, source)*Mxy
        
    if eij_name=='exz':
        e_xz_source, e_xz_source_z_der = read_strain_source_location(eij_name,nt0,nx,ny,nz,dz,Sloc)
        Mxz_der = 2*conv_eij_source(e_xz_source, source)
        CMT_z_der = CMT_z_der+2*conv_eij_source(e_xz_source_z_der, source)*Mxz      
        
    if eij_name=='eyz':
        e_yz_source, e_yz_source_z_der = read_strain_source_location(eij_name,nt0,nx,ny,nz,dz,Sloc)
        Myz_der = 2*conv_eij_source(e_yz_source, source)
        CMT_z_der = CMT_z_der+2*conv_eij_source(e_yz_source_z_der, source)*Myz

M0_new = M0 - 0.1*[[Mxx_der, Mxy_der, Mxz_der],[Mxy_der, Myy_der, Myz_der],[Mxz_der, Myz_der, Mzz_der]]
idepth = np.sign(CMT_z_der)

#Kappa_arr=Kappa.reshape(-1)
##GK = - np.multiply(sum1,Kappa_arr)
#GK =  np.multiply(sum1,Kappa_arr)
##sum2 = sumxx+sumyy+sumxxyy+2*(sumxy+sumxz+sumyz)        
#sum2 = 0.5*(sum_xy+sum_xz+sum_yz)        
#Mu_arr=Mu.reshape(-1)      
##GM = -2*np.multiply(sum2,Mu_arr)
#GM = 2*np.multiply(sum2,Mu_arr)
#
#GS=2*(GM-4/3*np.multiply(np.divide(Mu_arr,Kappa_arr),GK))
#GP=2*np.multiply(np.divide((Kappa_arr+4/3*Mu_arr),Kappa_arr),GK)	        
#     
#np.savetxt('KS.txt', GS)
#np.savetxt('KP.txt', GP)
