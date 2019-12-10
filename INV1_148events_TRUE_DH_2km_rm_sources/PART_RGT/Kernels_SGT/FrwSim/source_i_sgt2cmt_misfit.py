#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
"""
Created on Mon Dec  9 17:20:52 2019

@author: user

This script will calculate the CMT solution for an event directly from SGT data base and run simulation 
with the new CMT solution, calculate the misfit between the new simulated vs observed data and compare with 
the old misfit to see if the new CMT solution is actually better than the one given from GeoNet

"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy import fftpack
from scipy import signal
from qcore import timeseries
from scipy import integrate
from scipy.signal import butter, lfilter
from scipy.signal import resample

from numpy.linalg import inv
#import pdb; 
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def read_srf(fname):
    """
    Convinience function for reading files in the Graves and Pitarka format
    """
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    data = []

    for line in lines[7:]:
        data.append([float(val) for val in line.split()])

    data=np.concatenate(data) 
    
    line5=lines[5].split()
    line6=lines[6].split()   
#    num_pts=float(line1[0])
    dt_srf=float(line5[7])
    strike=float(line5[4])
    dip=float(line5[5])
    rake=float(line6[0])

    return data, dt_srf, strike, dip, rake

def Vsp_read(nx,ny,nz,snap_file):
    fid=open(snap_file,'r')
    sek=np.fromfile(fid, dtype='<f4')
    #print(sek.shape)
    Vp=np.reshape(sek,[ny,nz,nx])

    return Vp

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

def time_shift_emod3d(data,delay_Time,dt):
    n_pts = len(data)
    ndelay_Time = int(delay_Time/(dt))
    data_shift = np.zeros(data.shape)
    data_shift[0:n_pts-ndelay_Time] = data[ndelay_Time:n_pts]
    return data_shift

def read_sgt(matrix_file_fx):
    open_file = open(matrix_file_fx, "rb")
    geoproj = np.fromfile(open_file, dtype="i4", count=1)
    [modellon, modellat, modelrot, xshift, yshift] = np.fromfile(open_file, dtype="f4", count=5)
    [globnp, localnp, nt] = np.fromfile(open_file, dtype="i4", count=3)
    #print(localnp) 
    indx = np.zeros((localnp,1))
    coord_sgt = np.zeros((localnp,3))
    
    Mxx = np.zeros((localnp,nt))
    Myy = np.zeros((localnp,nt))
    Mzz = np.zeros((localnp,nt))
    
    Mxy = np.zeros((localnp,nt))
    Mxz = np.zeros((localnp,nt))
    Myz = np.zeros((localnp,nt))
    
    for ik in range(0,localnp):
        indx[ik] = np.fromfile(open_file, dtype="i8", count=1)
        coord_sgt[ik,:] = np.fromfile(open_file, dtype="i4", count=3)
        dh = np.fromfile(open_file, dtype="f4", count=1)
     
    for ik in range(0,localnp):
        Mxx[ik,:] = np.fromfile(open_file, dtype="f4", count=nt)
        Myy[ik,:] = np.fromfile(open_file, dtype="f4", count=nt)
        Mzz[ik,:] = np.fromfile(open_file, dtype="f4", count=nt)         
        
        Mxy[ik,:] = np.fromfile(open_file, dtype="f4", count=nt)
        Mxz[ik,:] = np.fromfile(open_file, dtype="f4", count=nt)
        Myz[ik,:] = np.fromfile(open_file, dtype="f4", count=nt) 
        
    return geoproj, modellon, modellat, modelrot, xshift, yshift, globnp, localnp, nt, \
           indx, coord_sgt, dh, Mxx, Myy, Mzz, Mxy, Mxz, Myz    
   
def search_coord(source, coord_sgt, n_coord):
    i_source=-1
    for ik in range(0, n_coord):
        #print(np.sum(np.square(np.array(coord_sgt[ik,:])-np.array(source))))
        if (np.sum(np.square(np.array(coord_sgt[ik,:])-np.array(source))))<3:
            i_source=ik
            break
            
    if(i_source==-1):
        print('no source in the list')
    
    return i_source
        
def rwm_wd(stat_data_0_Sf,stat_data_0_Of,num_pts, dt,wd):

    t = np.arange(num_pts)*dt
    stat_data_0_Sf_wd = np.multiply(stat_data_0_Sf,wd)
    stat_data_0_Of_wd = np.multiply(stat_data_0_Of,wd)

    rwm1_arr=np.square((stat_data_0_Sf_wd-stat_data_0_Of_wd))
    rwm2_arr=np.square((stat_data_0_Sf_wd))
    rwm3_arr=np.square((stat_data_0_Of_wd))

    rwm1=integrate.simps(rwm1_arr,t)
    rwm2=integrate.simps(rwm2_arr,t)
    rwm3=integrate.simps(rwm3_arr,t)

#    rwm1 = np.sum(np.cumsum(rwm1_arr)*dt)
#    rwm2 = np.sum(np.cumsum(rwm2_arr)*dt)
#    rwm3 = np.sum(np.cumsum(rwm3_arr)*dt)    

    err_rwm=rwm1/((rwm2*rwm3)**0.5)

    return err_rwm
    
# Read_Dev_strain_ex2.m
nx=65
ny=65
#nz=75
nz=40
nts=2000
nt0=400
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
delay_Time=(3/flo)

station_file = '../../../StatInfo/STATION.txt'  
nRec, R, statnames = read_stat_name(station_file)

source_file = '../../../StatInfo/SOURCE.txt'
nShot, S, sNames = read_source_new(source_file)

with open('statnames_RGT.txt', 'r') as f:
    lines = f.readlines()    
    statnames_RGT = [] 
    for line in lines:
        line_i=line.split()
        statnames_RGT.append(line_i[0])

print(statnames_RGT)

Nr=len(statnames_RGT)
GV=['AAA_fx.sgt','AAA_fy.sgt','AAA_fz.sgt']

################################Store the RGT matrix
#AA=np.zeros([Ns,Nr,3,nt0,6])
AA=np.zeros([Nr,3,nts,9])
#ij_array=[0, 1, 2, 4, 5, 8]    
M_all=np.zeros([1,9])

fi=open('iShot.dat','r')
ishot=np.int64(np.fromfile(fi,dtype='int64'))
fi.close()
ishot=int(ishot)

for i,statname in enumerate(statnames_RGT):
    for k in range(0,3):
        fw_file='../Dev_Strain/'+statname+'/'+GV[k]
        _, _, _, _, _, _, _, localnp, nt, indx, coord_sgt, dh, Mxx, Myy, Mzz, Mxy, Mxz, Myz = read_sgt(fw_file)
        print(k)        
        
        #if (k==0):
        source = S[ishot-1,:] 
        #print(source)
        #print(coord_sgt)
        i_source = search_coord(source, coord_sgt, localnp)
        print(i_source)

        AA[i,k,:,0] = time_shift_emod3d(Mxx[i_source,:],delay_Time,dt)
        AA[i,k,:,4] = time_shift_emod3d(Myy[i_source,:],delay_Time,dt)
        AA[i,k,:,8] = time_shift_emod3d(Mzz[i_source,:],delay_Time,dt)
        
        AA[i,k,:,1] = time_shift_emod3d(Mxy[i_source,:],delay_Time,dt)
        AA[i,k,:,3] = time_shift_emod3d(Mxy[i_source,:],delay_Time,dt)
        
        AA[i,k,:,2] = time_shift_emod3d(Mxz[i_source,:],delay_Time,dt)
        AA[i,k,:,6] = time_shift_emod3d(Mxz[i_source,:],delay_Time,dt)
        
        AA[i,k,:,5] = time_shift_emod3d(Myz[i_source,:],delay_Time,dt)
        AA[i,k,:,7] = time_shift_emod3d(Myz[i_source,:],delay_Time,dt)           
                                          
np.savetxt('AA_SGT.txt',AA.reshape(-1))
#input('-->')
##############################################

R_Time_record_arr = np.loadtxt('../R_Time_record_148s_dh_2km.txt') 
R_Time_record = R_Time_record_arr.reshape([2,nShot,nRec])

GV=['.090','.000','.ver']

Err=0
   
#    os.system('cp Vel_ob_new_V2/Vel_ob_'+str(ishot)+'/*.* /home/user/workspace/GMPlots/Sim/Vel_ob/')
#    os.system('cp Vel_ob_ref_01Hz/Vel_ob_'+str(ishot)+'/*.* /home/user/workspace/GMPlots/Sim/Vel_es/')    

#    mainfolder='/home/user/workspace/GMPlots/Sim/Vel_es/'
#    mainfolder_o='/home/user/workspace/GMPlots/Sim/Vel_ob/'

#mainfolder='Vel_opt/Vel_ob_'+str(ishot)+'/'
mainfolder='../Vel_es/Vel_es_'+str(ishot)+'/'
#mainfolder_o='../../../Kernels/Vel_ob_new_V2/Vel_ob_'+str(ishot)+'/'
mainfolder_o='../../../Kernels/Vel_ob_ref/Vel_ob_'+str(ishot)+'/'

obs=np.zeros(int(3*Nr*nt))
AA_ishot=AA
AA_ishot=np.reshape(AA_ishot,[Nr*3*nt,9])
AA_max=100*np.max(np.abs(AA_ishot))

ik=0
for i,statname in enumerate(statnames):
    if(statname in statnames_RGT):
        #distance=((R[i,1]-S[ishot-1,1])**2+(R[i,2]-S[ishot-1,2])**2+(R[i,0]-S[ishot-1,0])**2)**(0.5)
        print(statname)
        for k in range(0,3):
            s0=statname+GV[k]

            stat_data_0_S_org  = timeseries.read_ascii(mainfolder+s0)
            stat_data_0_S = time_shift_emod3d(stat_data_0_S_org,delay_Time,dt)
            stat_data_0_S  = np.multiply(signal.tukey(int(num_pts),0.1),stat_data_0_S)

            stat_data_0_O  = timeseries.read_ascii(mainfolder_o+s0)
            stat_data_0_O  = np.multiply(signal.tukey(int(num_pts),0.1),stat_data_0_O)
            #stat_data_0_S = signal.detrend(stat_data_0_S)             
            #stat_data_0_O = signal.detrend(stat_data_0_O)

    #        stat_data_0_S = signal.filtfilt(b, a, stat_data_0_S)           
    #        stat_data_0_O = signal.filtfilt(b, a, stat_data_0_O)

            stat_data_0_S = butter_bandpass_filter(stat_data_0_S, lowcut, highcut, fs, order=4)
            stat_data_0_O = butter_bandpass_filter(stat_data_0_O, lowcut, highcut, fs, order=4)

            wd=np.zeros(stat_data_0_S.shape)    
            wd[int(R_Time_record[0,ishot-1,i]/dt):int(R_Time_record[1,ishot-1,i]/dt)+1]=1     
            wd=np.ones(wd.shape)

            #stat_data_0_S = np.multiply(stat_data_0_S,wd)
            #stat_data_0_O = np.multiply(stat_data_0_O,wd) 
            
            Err = Err+rwm_wd(stat_data_0_S,stat_data_0_O,num_pts, dt,wd)
            
            stat_data_0_S = np.cumsum(stat_data_0_S)*dt          
            stat_data_0_O = np.cumsum(stat_data_0_O)*dt                     
            
#           obs[nt0*k*ik+nt0*k:nt0*k*ik+nt0*(k+1)]=resample(stat_data_0_S,int(nt0)) 
            obs[nt*k*ik+nt*k:nt*k*ik+nt*(k+1)]=-(stat_data_0_S)

#            if(k<2):
#                obs[nt*k*ik+nt*k:nt*k*ik+nt*(k+1)] = stat_data_0_S
#            else:
#                obs[nt*k*ik+nt*k:nt*k*ik+nt*(k+1)] = -stat_data_0_S

        ik=ik+1
    #Add iso(M0)=0 to A and d
#    d=np.r_[d, 0]
    obs=np.r_[obs,[0,0,0,0]]
    b = np.zeros((1,9)); b[0,0] = 1; b[0,4] = 1; b[0,8] = 1;  b = b*AA_max;
    AA_ishot = np.r_[AA_ishot, b]    
    #Add symmetriy constraint
#    d=np.r_[d, 0]; d=np.r_[d, 0]; d=np.r_[d, 0];
    
    b = np.zeros((1,9)); b[0,1] = 1; b[0,3] = -1; b = b*AA_max;
    AA_ishot = np.r_[AA_ishot, b] 

    b = np.zeros((1,9)); b[0,2] = 1; b[0,6] = -1; b = b*AA_max;
    AA_ishot = np.r_[AA_ishot, b] 

    b = np.zeros((1,9)); b[0,5] = 1; b[0,7] = -1; b = b*AA_max;
    AA_ishot = np.r_[AA_ishot, b]     
    #Inversion
    AA_inv=inv(np.matmul(np.transpose(AA_ishot),AA_ishot))     
    AA_d=np.matmul(np.transpose(AA_ishot),obs)
    M_ishot=np.matmul(AA_inv,AA_d)
    
M_ishot_array=M_ishot.reshape(-1)
np.savetxt('CMT_inv_1events_nstats.txt',M_ishot_array)
    
print('Err='+str(Err))
f_err = open('cmt_err_srf.dat','w')            
            

