#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
import numpy as np
#import matplotlib.pyplot as plt
#import pdb; 
#import multiprocessing as mp
#from multiprocessing import Pool
from qcore import timeseries
from scipy.signal import resample

def time_shift_emod3d(data,delay_Time,dt):
    n_pts = len(data)
    ndelay_Time = int(delay_Time/(dt))
    data_shift = np.zeros(data.shape)
    data_shift[0:n_pts-ndelay_Time] = data[ndelay_Time:n_pts]
    return data_shift

def conv_eij_source(e_ij_source, source):
   
#    x = np.arange(0,nt,1)
#    pool = mp.Pool(processes=4)
#    arr_dt_sumxy=pool.map(conv_wf_it_2,x)
#    pool.close()
#    pool.join()   
    nt = len(source)
    sum_ij=0
    
    for it in np.arange(0,nt,1):    
        sum_ij=sum_ij+e_ij_source[it]*dt*source[it]
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

source_file = '../../../../StatInfo/SOURCE.txt'
nShot, S, sNames = read_source_new(source_file)

fi=open('iShot.dat','r')
iShot=np.int64(np.fromfile(fi,dtype='int64'))
fi.close()
iShot=int(iShot)
Sloc =S[iShot-1,:]

#source = np.zeros([nt0,1])
#stf_source = timeseries.read_ascii('Stf/fwd01_2tri-p10-h2010.00')
stf_source = timeseries.read_ascii('Stf/reg0_cos1.00')
source = resample(stf_source,nt0)

fi1=open('iter_s.dat','r')
iter_s=np.int64(np.fromfile(fi1,dtype='int64'))
fi1.close()
iter_s=int(iter_s)
print('iter_s='+str(iter_s))
#M0=np.zeros([3,3])
M0_arr=np.loadtxt('Dump_source/M0.it'+str(iter_s)+'.txt')
M0=M0_arr.reshape([3,3])
    

Mxx=M0[0,0];Myy=M0[1,1];Mzz=M0[2,2];
Mxy=M0[0,1];Mxz=M0[0,2];Myz=M0[1,2];

with open('../../Dev_Strain/adj01_xyzts.exx', 'rb') as fid:
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

M_der=np.array([[Mxx_der, Mxy_der, Mxz_der],[Mxy_der, Myy_der, Myz_der],[Mxz_der, Myz_der, Mzz_der]])
grad_M_arr = M_der.reshape(-1)
#gra1=gra
if (iter_s==0):
    gra_M=M_der
    grad_M_arr1=grad_M_arr
    beta_N=0
    beta_D=0
#    
if (iter_s>0):#CG Polak-Ribiere
    grad_M_last_arr=np.loadtxt('Dump_source/Grad_M.it'+str(iter_s-1)+'.txt')
    beta_N=np.dot(grad_M_arr,grad_M_arr-grad_M_last_arr)
    beta_D=np.dot(grad_M_arr,grad_M_last_arr)
    print('beta_N='+str(beta_N)+'and beta_D='+str(beta_D))
    grad_M_arr1=grad_M_arr+beta_N/beta_D*grad_M_last_arr
    gra_M=grad_M_arr1.reshape([3,3])
    
np.savetxt('Dump_source/Grad_M.it'+str(iter_s)+'.txt',grad_M_arr1)
beta=[beta_N, beta_D]
np.savetxt('Dump_source/Beta.it'+str(iter_s)+'.txt', beta)

#M0_new = M0 - 0.05*LA.norm(M0)/LA.norm(gra_M)*gra_M
#np.savetxt('Dump_source/M0.iter'+str(iter_s+1),M0_new.reshape(-1))

idepth = np.sign(CMT_z_der)

#print('M0='+str(M0))
print('M_der='+str(M_der))
print(idepth)

