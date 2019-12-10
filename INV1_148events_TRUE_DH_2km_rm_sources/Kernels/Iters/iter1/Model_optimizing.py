#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 11:24:47 2019

@author: user
"""
import numpy as np
import os
#import time
#from numpy.linalg import solve
#from qcore import timeseries as ts

def Vsp_read(nx,ny,nz,snap_file):
    fid=open(snap_file,'r')
    sek=np.fromfile(fid, dtype=np.float32)
    Vp=np.reshape(sek,[ny,nz,nx])
    return Vp

def write_model_Vsp(rho,Vs,Vp):
    [ny,nz,nx]=Vs.shape
    
    model_file='rho3dfile1.d'
    fid=open(model_file,'wb')
    sek1 = np.array(rho, dtype=np.float32)
    sek1.astype('float32').tofile(fid)
    
    model_file0='vs3dfile1.s'
    model_file1='vp3dfile1.p'

    fid00=open(model_file0,'wb')
    sek2 = np.array(Vs, dtype=np.float32)
    sek2.astype('float32').tofile(fid00)

    fid01=open(model_file1,'wb')
    sek3 = np.array(Vp, dtype=np.float32)
    sek3.astype('float32').tofile(fid01)

###############Main script#############
nx=269
ny=269
nz=68
nts=10000
nt=500

#dx=1.0
#dy=dx
#dz=dx
#dt=0.02
##############################################################################
fi1=open('iNumber.dat','r')
it=int(np.fromfile(fi1,dtype='int64'))
fi1.close()
print('iNumber='+str(it))

########################################
fi1=open('st.dat','r')
st=np.fromfile(fi1,dtype=np.float64)
fi1.close()
print('st='+str(st))

snap_file1='../../../Model/Models/vs3dfile_in.s'
snap_file2='../../../Model/Models/vp3dfile_in.p'
snap_file3='../../../Model/Models/rho3dfile_in.d'
Vs=Vsp_read(nx,ny,nz,snap_file1)
Vp=Vsp_read(nx,ny,nz,snap_file2)
rho=Vsp_read(nx,ny,nz,snap_file3)
print('Dump/Grads_iter_'+str(it)+'.txt')    
Gra_S_arr = np.loadtxt('Dump/Grads_iter_'+str(it)+'.txt')   
Gra_P_arr = np.loadtxt('Dump/Gradp_iter_'+str(it)+'.txt')  

Vsp=np.zeros(2*nx*ny*nz)
grad_Vsp=np.zeros(2*nx*ny*nz)

Vsp[0:nx*ny*nz] = Vs.reshape(-1)
Vsp[nx*ny*nz:2*nx*ny*nz] = Vp.reshape(-1)

grad_Vsp[0:nx*ny*nz] = Gra_S_arr
grad_Vsp[nx*ny*nz:2*nx*ny*nz] = Gra_P_arr

st_Vsp0=st*np.max(np.abs(Vsp))/np.max(np.abs(grad_Vsp))
        
Vsp0=Vsp-st_Vsp0*grad_Vsp
Vs0=np.reshape(Vsp0[0:nx*ny*nz],[ny,nz,nx])
Vp0=np.reshape(Vsp0[nx*ny*nz:2*nx*ny*nz],[ny,nz,nx])

print('Mu21='+str(st))	
print('max st_Vp0*Gra_Vp ='+str(np.max(np.abs(st_Vsp0*grad_Vsp))))  
write_model_Vsp(rho,Vs0,Vp0)
#time.sleep(5)
os.system('mv vs3dfile1.s Model/vs3dfile_opt.s')
os.system('mv vp3dfile1.p Model/vp3dfile_opt.p')
os.system('mv rho3dfile1.d Model/rho3dfile_opt.d')
