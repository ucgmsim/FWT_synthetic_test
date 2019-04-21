#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
import numpy as np
#import matplotlib.pyplot as plt
#import pdb; 
#import multiprocessing as mp
#from multiprocessing import Pool

def conv_wf_it(it):
    """
         Convolve forward and backward
    """   
    ef_i=ef[it*nx*ny*nz:(it+1)*nx*ny*nz]
    eb_i=eb[(nt-it-1)*nx*ny*nz:(nt-it)*nx*ny*nz]

    dt_sum=np.zeros(ef_i.shape)
    for iy in np.arange(0,ny,1):
        dt_sum[iy*nx*nz:(iy+1)*nx*nz] = np.multiply(ef_i[iy*nx*nz:(iy+1)*nx*nz],eb_i[iy*nx*nz:(iy+1)*nx*nz])    
#    print('it='+str(it))
    return dt_sum

def conv_wf(dt):
   
#    x = np.arange(0,nt,1)
#    pool = mp.Pool(processes=4)
#    arr_dt_sumxy=pool.map(conv_wf_it_2,x)
#    pool.close()
#    pool.join()   
    sum_ij=conv_wf_it(0)
    for it in np.arange(0,nt,1):    
        sum_ij=sum_ij+dt*conv_wf_it(it)           
#    print('done'+eij_name)    
        
    return sum_ij

def Vsp_read(nx,ny,nz,snap_file):
    fid=open(snap_file,'r')
    sek=np.fromfile(fid, dtype='<f4')
    print(sek.shape)
    Vp=np.reshape(sek,[ny,nz,nx])
    return Vp
    
def read_strain(eij_name):
    
    matrix_file_fw=fw_file+eij_name
    matrix_file_bw=bw_file+eij_name
    
    fid1=open(matrix_file_fw, 'rb')
    matrix_dummy_fw=np.fromfile(fid1, np.float32)
    matrix_dummy_fw=matrix_dummy_fw[15:]
    
    fid2=open(matrix_file_bw, 'rb')
    matrix_dummy_bw=np.fromfile(fid2, np.float32)  
    matrix_dummy_bw=matrix_dummy_bw[15:]
    
    return matrix_dummy_fw, matrix_dummy_bw

# Read_Dev_strain_ex2.m
nx=200
ny=200
nz=75
nts=5000
nt=500
nxyz=nx*ny*nz

with open('../../Dev_Strain/fwd01_xyzts.exx', 'rb') as fid:
    data_array = np.fromfile(fid, np.float32)
    data_array=data_array[0:14]
    dx=data_array[8]
    dy=data_array[9]
    dz=data_array[10]
    dt=data_array[11]
    del data_array
    
snap_file1='../../../Model/Models/vs3dfile_in.s'
snap_file2='../../../Model/Models/vp3dfile_in.p'
snap_file3='../../../Model/Models/rho3dfile_in.d'

Vs=Vsp_read(nx,ny,nz,snap_file1)
Vp=Vsp_read(nx,ny,nz,snap_file2)
rho=Vsp_read(nx,ny,nz,snap_file3)

Lamda=np.multiply(rho,(np.multiply(Vp,Vp)-2*np.multiply(Vs,Vs)));
Mu=np.multiply(rho,np.multiply(Vs,Vs))
Kappa=Lamda+2/3*Mu

#mainfolder_fig='/home/user/workspace/GMPlots/Figs/'
eijp_names = {'exx','eyy','ezz','exy','exz','eyz'};
#eijp_names = {'exyp','exzp','eyzp'}
fw_file='../../Dev_Strain/fwd01_xyzts.';
bw_file='../../Dev_Strain/adj01_xyzts.';

for eij_name in eijp_names:
    print(eij_name)
    if eij_name=='exx':
        exx3, exxb = read_strain(eij_name)
        
    if eij_name=='eyy':
        eyy3, eyyb = read_strain(eij_name)

    if eij_name=='ezz':
        ezz3, ezzb = read_strain(eij_name)
        
    if eij_name=='exy':
        ef,eb = read_strain(eij_name)
        sum_xy=4*conv_wf(dt)
        
    if eij_name=='exz':
        ef,eb = read_strain(eij_name)        
        sum_xz=4*conv_wf(dt)             
        
    if eij_name=='eyz':
        ef,eb = read_strain(eij_name)        
        sum_yz=4*conv_wf(dt)


ef = exx3+eyy3+ezz3
eb = exxb+eyyb+ezzb    
sum1 = conv_wf(dt)

Kappa_arr=Kappa.reshape(-1)
GK = - np.multiply(sum1,Kappa_arr)
#sum2 = sumxx+sumyy+sumxxyy+2*(sumxy+sumxz+sumyz)        
sum2 = 0.5*(sum_xy+sum_xz+sum_yz)        
Mu_arr=Mu.reshape(-1)      
GM = -2*np.multiply(sum2,Mu_arr)
     
GS=2*(GM-4/3*np.multiply(np.divide(Mu_arr,Kappa_arr),GK))
GP=2*np.multiply(np.divide((Kappa_arr+4/3*Mu_arr),Kappa_arr),GK)	        
     
np.savetxt('KS.txt', GS)
np.savetxt('KP.txt', GP)
