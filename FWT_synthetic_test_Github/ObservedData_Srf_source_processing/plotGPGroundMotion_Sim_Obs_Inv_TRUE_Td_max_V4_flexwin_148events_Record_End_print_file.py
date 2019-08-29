#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:43:17 2018
    
"""
#import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy import fftpack
from scipy import signal
from qcore import timeseries
from scipy import integrate
from scipy.signal import butter, lfilter
#from statistics import median

def read_flexwin(filename):
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    if(len(lines)>2):
        line2=lines[2].split()
        t_on=float(line2[1])
        t_off=float(line2[2])
        td_shift=float(line2[3])
        cc=float(line2[4])
        
    else:
        t_on = 0; t_off = 0; td_shift = 0; cc = 0;       
    
    return t_on, t_off, td_shift, cc

def winpad(lt,t_off,t_on,pad):
    """
    Sin taper window
    """    
    #pad=5
    L=t_off-t_on+2*pad
    
#    window=signal.gaussian(30, 4)
#    window1=signal.resample(window,L,axis=0, window=None)
#    window=np.linspace(1, 1, L)))
    window=np.ones((L))
    
    #x=np.arange(0,pad,1)
    x=np.linspace(0, np.pi/2, pad)
    sinx=np.sin(x)
    window[0:pad] = sinx
    window[L-pad:L] = sinx[::-1]    
    print('lt='+str(lt))    
    ar1=np.zeros((t_on-pad))
    ar2=np.zeros((lt-t_off-pad))
    window_pad0 = np.concatenate((ar1,window))
    window_pad = np.concatenate((window_pad0,ar2))    
    
    return window_pad  
############################################################

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


def readGP(loc, fname):
    """
    Convinience function for reading files in the Graves and Pitarka format
    """
    with open("/".join([loc, fname]), 'r') as f:
        lines = f.readlines()
    
    data = []

    for line in lines[2:]:
        data.append([float(val) for val in line.split()])

    data=np.concatenate(data) 
    
    line1=lines[1].split()
    num_pts=float(line1[0])
    dt=float(line1[1])
    shift=float(line1[4])

    return data, num_pts, dt, shift

def readGP_2(loc, fname):
    """
    Convinience function for reading files in the Graves and Pitarka format
    """
    with open("/".join([loc, fname]), 'r') as f:
        lines = f.readlines()
    
    data = []

    for line in lines[2:]:
        data.append([float(val) for val in line.split()])

    data=np.concatenate(data) 
    
    line1=lines[1].split()
    num_pts=float(line1[0])
    dt=float(line1[1])
    shift=float(line1[4])

    return data, num_pts, dt, shift

def adjust_for_time_delay(ts, dt, shift):
    """
        ts: time series data
    """
    t0_index = int(shift/dt)
    if t0_index == 0:
        num_pts = ts.size
    elif t0_index > 0:
        ts = np.concatenate((np.zeros(t0_index), ts))
        num_pts = ts.size
    elif t0_index <0:
        ts = ts[np.abs(t0_index):]
        num_pts = ts.size

    return ts, num_pts, dt

def get_adjusted_stat_data(loc, stat_code):
    """
    Time series data is adjust for time shift

    returns a dictionary with components:
        000, 090, ver, t, stat_code
    """
    stat_data = {"000": None, "090": None, "ver":None, "t": None, "name": None}
    g=981. #cm/s^2
    stat_data["000"], num_pts, dt, shift  = readGP(loc, ".".join([stat_code, "000"]))
    stat_data["090"], num_pts, dt, shift = readGP(loc, ".".join([stat_code, "090"]))   
    stat_data["ver"], num_pts, dt, shift = readGP(loc, ".".join([stat_code, "ver"]))


    stat_data["000"], num_pts, dt = adjust_for_time_delay(stat_data["000"], dt, shift)
    stat_data["090"], num_pts, dt = adjust_for_time_delay(stat_data["090"], dt, shift)
    stat_data["ver"], num_pts, dt = adjust_for_time_delay(stat_data["ver"], dt, shift)

    t = np.arange(num_pts)*dt
    stat_data["t"] = t

    stat_data["name"]=stat_code
    return stat_data

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

#def butter_bandpass(lowcut, highcut, fs, order):
#    nyq = 0.5 * fs
#    low = lowcut / nyq
#    high = highcut / nyq
#    b, a = butter(order, [low, high], btype='band')
#    return b, a
#
#
#def butter_bandpass_filter(data, lowcut, highcut, fs, order):
#    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#    y = lfilter(b, a, data)
#    return y
def rms(stat_data):

    num_pts=len(stat_data)
    D = (np.sum(np.square(stat_data))/num_pts)**0.5

    return stat_data/D

#def winpad(num_pts,I_max):
#    """
#    Sin taper window
#    """
#    pad=5
#    t_off=I_max+268;
#    if I_max>275:
#        t_on=I_max-268
#    else:
#        t_on=10
#            
#    L=t_off-t_on+2*pad
#
#    window=np.ones((L))
#
#    #x=np.arange(0,pad,1)
#    x=np.linspace(0, np.pi/2, pad)
#    sinx=np.sin(x)
#    window[0:pad] = sinx
#    window[L-pad:L] = sinx[::-1]
##    print('lt='+str(num_pts))    
#    ar1=np.zeros((t_on-pad))
#    ar2=np.zeros((num_pts-t_off-pad))
#    window_pad0 = np.concatenate((ar1,window))
#    window_pad = np.concatenate((window_pad0,ar2))
#
#    return window_pad

def rwm_wd(stat_data_0_Sf,stat_data_0_Of,num_pts, dt,wd):
#    rwm1=0
#    rwm2=0
#    rwm3=0
    t = np.arange(num_pts)*dt
    stat_data_0_Sf = np.multiply(stat_data_0_Sf,wd)
    stat_data_0_Of = np.multiply(stat_data_0_Of,wd)

    rwm1_arr=np.square((stat_data_0_Sf-stat_data_0_Of))
    rwm2_arr=np.square((stat_data_0_Sf))
    rwm3_arr=np.square((stat_data_0_Of))

    rwm1=integrate.simps(rwm1_arr,t)
    rwm2=integrate.simps(rwm2_arr,t)
    rwm3=integrate.simps(rwm3_arr,t)

#    rwm1 = np.sum(np.cumsum(rwm1_arr[wd1:wd2])*dt)
#    rwm2 = np.sum(np.cumsum(rwm2_arr[wd1:wd2])*dt)
#    rwm3 = np.sum(np.cumsum(rwm3_arr[wd1:wd2])*dt)    

    err_rwm=rwm1/((rwm2*rwm3)**0.5)
    
#    if(err_rwm>8):
#        err_rwm=8    

    return err_rwm

#def ncc(stat_data_0_Sf,stat_data_0_Of,num_pts, delta_T,dt):
#    """
#    Normalized correlation coefficient
#    """    
##    rwm1=0
##    rwm2=0
##    rwm3=0
#    num_delta_t = int(delta_T/dt);    td = np.arange(-num_delta_t,num_delta_t)*dt;
#    t = np.arange(num_pts)*dt
#    ncc_array=np.zeros(len(td))
#    
#    for it in range(0,len(td)):
#        stat_data_0_S_shift = np.zeros(num_pts); n_shift=int(np.abs((td[it]/dt)));
#        
#        if td[it]<0:
#            stat_data_0_S_shift[n_shift:num_pts] = stat_data_0_Sf[0:num_pts-n_shift]                             
#        else:
#            stat_data_0_S_shift[0:num_pts-n_shift] = stat_data_0_Sf[n_shift:num_pts]                     
#        
#        rwm1_arr=np.multiply(stat_data_0_S_shift,stat_data_0_Of)
#        rwm2_arr=np.square((stat_data_0_S_shift))
#        rwm3_arr=np.square((stat_data_0_Of))        
#        
#        rwm1=integrate.simps(rwm1_arr,t)
#        rwm2=integrate.simps(rwm2_arr,t)
#        rwm3=integrate.simps(rwm3_arr,t)
#    
#        ncc_array[it]=rwm1/((rwm2*rwm3)**0.5)
#    
#    ncc_max = np.max(ncc_array); id_max = np.argmax(ncc_array); td_max = td[id_max];
#    
#    return ncc_max, td_max   

def ncc2(stat_data_0_Sf,stat_data_0_Of,num_pts, delta_T,dt):
    """
    Normalized correlation coefficient
    """    
#    rwm1=0
#    rwm2=0
#    rwm3=0
    num_delta_t = int(delta_T/dt);    td = np.arange(-num_delta_t,num_delta_t)*dt;
#    t = np.arange(num_pts)*dt
    ncc_array=np.zeros(len(td))
    
    for it in range(0,len(td)):
        stat_data_0_O_shift = np.zeros(num_pts); n_shift=int(np.abs((td[it]/dt)));
        
#        if td[it]<0:
#            stat_data_0_O_shift[n_shift:num_pts] = stat_data_0_Of[0:num_pts-n_shift]                             
#        else:
#            stat_data_0_O_shift[0:num_pts-n_shift] = stat_data_0_Of[n_shift:num_pts]         

        if td[it]<0:
            stat_data_0_O_shift[0:num_pts-n_shift] = stat_data_0_Of[n_shift:num_pts]                  
        else:
            stat_data_0_O_shift[n_shift:num_pts] = stat_data_0_Of[0:num_pts-n_shift]
        
        rwm1_arr=np.multiply(stat_data_0_Sf,stat_data_0_O_shift)
        rwm2_arr=np.square((stat_data_0_Sf))
        rwm3_arr=np.square((stat_data_0_O_shift))        
        
#        rwm1=integrate.simps(rwm1_arr,t)
#        rwm2=integrate.simps(rwm2_arr,t)
#        rwm3=integrate.simps(rwm3_arr,t)
        rwm1 = np.sum(rwm1_arr)
        rwm2 = np.sum(rwm2_arr)
        rwm3 = np.sum(rwm3_arr)
    
        ncc_array[it]=rwm1/((rwm2*rwm3)**0.5)
    
    ncc_max = np.max(ncc_array); id_max = np.argmax(ncc_array); td_max = td[id_max];
    
    return ncc_max, td_max

def normpdf_python(x, mu, sigma):
   return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1*(x-mu)**2/2*sigma**2)

def TauP(stat_data_S,stat_data_O,num_pts, dt,wd):
    t = np.arange(num_pts)*dt
   
    ts=np.flip(-t[1:], axis=0)
    lTime = np.concatenate((ts,t), axis=0)
    w = normpdf_python(lTime, 0, 0.05)
    wp = w/max(w)

    Dis_S = np.multiply(stat_data_S,wd)
    Dis_O = np.multiply(stat_data_O,wd)
    
    corr=np.correlate(Dis_O,Dis_S,"full")
    wx=np.multiply(wp,corr)
    
    In=np.argmax(wx)
    TauP=lTime[In]
    
#    if (TauP>10):
#        TauP = 10
#
#    if (TauP<-10):
#        TauP = -10        
    
    return TauP

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

def time_shift(data,t_shift,dt):
    n_pts = len(data)
    nshift_T = int(t_shift/(dt))
    data_shift = np.zeros(data.shape)
    data_shift[nshift_T:n_pts] = data[0:n_pts-nshift_T]
    return data_shift

def time_shift_emod3d(data,delay_Time,dt):
    n_pts = len(data)
    ndelay_Time = int(delay_Time/(dt))
    data_shift = np.zeros(data.shape)
    data_shift[0:n_pts-ndelay_Time] = data[ndelay_Time:n_pts]
    return data_shift
##############################################################
#orig_stdout = sys.stdout
#f = open('output.txt','w')
#sys.stdout = f    

station_file = '/home/user/workspace/GMPlots/Christchurch_Events/INV1_148events_TRUE_DH_2km/Kernels/STATION.txt'  
nRec, R, statnames = read_stat_name(station_file)
#nShot=4;ishot=99;
source_file = '/home/user/workspace/GMPlots/Christchurch_Events/INV1_148events_TRUE_DH_2km/Kernels/SOURCE.txt'
nShot, S, sNames = read_source_new(source_file)

R_ishot_new=np.ones(nRec)
#R_all=np.reshape(R_arr,[nShot,nRec])

#statnames = ['CACS','CBGS','CMHS','CRLZ','DFHS','NNBS','PPHS','REHS']   
print('statnames')
print(statnames)

num_pts=2000; dt=0.08;
t = np.arange(num_pts)*dt


############/nesi/nobackup/nesi00213/RunFolder/tdn27/rgraves/Adjoint/Syn_VMs/Kernels/#########################
fs = 1/dt
lowcut = 0.05
#highcut = 0.4
highcut = 0.1

fc = highcut  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(4, w, 'low')

R_arr=np.loadtxt('/home/user/workspace/GMPlots/Christchurch_Events/INV1_148events_TRUE_DH_2km/Kernels/R_all_148s_dh_2km.txt')
R_all=np.reshape(R_arr,[nShot,nRec])

#delta_T=10
delta_T=5
flo=0.1
delay_Time=(3/flo)
t_shift=0
nShot=148

index_all=np.zeros((nRec,3,nShot))
index_stat=np.zeros(nShot)
index_es=np.zeros(nShot)
index_e=np.zeros(nRec)

ncc_all=np.zeros((nRec,3,nShot))
td_max_all=np.zeros((nRec,3,nShot))
tp_max_all=np.zeros((nRec,3,nShot))
index_all=np.zeros((nRec,3,nShot))
all_stat=0

rwn_max_count=[]  
cc_max_count=[] 
ncc_max_count=[] 
td_max_count=[] 
tp_max_count=[] 
win_max_count=[]
dist_max_count=[]

count_t_end_corrected=0

TauP_Ns=np.zeros(nShot)
TauP_Nr=np.zeros(nRec)

#R_Time_record_arr = np.loadtxt('R_Time_record_148s_dh_2km.txt')  
R_Time_record_arr = np.loadtxt('R_Time_record_148s_dh_2km_num_pts_V2.txt') 
R_Time_record = R_Time_record_arr.reshape([2,nShot,nRec])

GV=['.090','.000','.ver']
for ishot in range(1,nShot+1):
#for ishot in range(1,1+1):
    
    if (S[ishot-1,2]<11):
#    if (S[ishot-1,2]<20) and (np.abs(ishot-26)>0) and (np.abs(ishot-49)>0) and (np.abs(ishot-51)>0):    
        os.system('cp Vel_ob_V2/Vel_ob_'+str(ishot)+'/*.* /home/user/workspace/GMPlots/Sim/Vel_ob/')
#        os.system('cp Vel_ob_ref/Vel_ob_'+str(ishot)+'/*.* /home/user/workspace/GMPlots/Sim/Vel_es/')
#        os.system('cp Vel_es_inv_2/Vel_ob_'+str(ishot)+'/*.* /home/user/workspace/GMPlots/Sim/Vel_es/')  
        
        os.system('cp Vel_ob_ref_01Hz/Vel_ob_'+str(ishot)+'/*.* /home/user/workspace/GMPlots/Sim/Vel_es/')    
#        os.system('cp Vel_opt_flexwin/Vel_ob_'+str(ishot)+'/*.* /home/user/workspace/GMPlots/Sim/Vel_es/')      
       
        ncc_R=np.zeros((nRec,3)); td_max_R=np.zeros((nRec,3));
    
        stat_count=[]
    #    num_count=[]
             
        for i,statname in enumerate(statnames):
            if (R_all[ishot-1,i]==1):
#                stat_count.append(statname)
#                num_count.append(i)
                distance=((R[i,1]-S[ishot-1,1])**2+(R[i,2]-S[ishot-1,2])**2+(R[i,0]-S[ishot-1,0])**2)**(0.5) *2
                
                plt.figure(figsize=(15,1.25))  
                for k in range(0,3):
#                    k=2-ki
                    s0=statname+GV[k]
                    gmdata_obs_vi_O = timeseries.read_ascii('/home/user/workspace/GMPlots/Sim/Vel_ob/'+s0)
                    
                    gmdata_sim_org = timeseries.read_ascii('/home/user/workspace/GMPlots/Sim/Vel_es/'+s0)           
                    gmdata_sim_vi_O = time_shift_emod3d(gmdata_sim_org,delay_Time,dt)
                    
#                    gmdata_obs_vi_O = time_shift(gmdata_obs_vi_O,t_shift,dt)
#                    gmdata_sim_vi_O = time_shift(gmdata_sim_vi_O,t_shift,dt)

                    ylimit=10
                    yflimit=2500
                    
        #            vxyz='vx'
                    vxyz=GV[k]
                           
        #            gmdata_obs_vi = signal.filtfilt(b, a, gmdata_obs_vi_O)               
        #            gmdata_sim_vi = signal.filtfilt(b, a, gmdata_sim_vi_O)
        #            gmdata_sim_vi1 = signal.filtfilt(b, a, gmdata_sim_vi1_O)   

                    gmdata_obs_vi = butter_bandpass_filter(gmdata_obs_vi_O, lowcut, highcut, fs, order=4)        
                    gmdata_sim_vi = butter_bandpass_filter(gmdata_sim_vi_O, lowcut, highcut, fs, order=4)  
#                    gmdata_sim_vi1 = butter_bandpass_filter(gmdata_sim_vi1_O, lowcut, highcut, fs, order=4)  
                    
#                    gmdata_obs_vi  = np.multiply(signal.tukey(int(num_pts),1),gmdata_obs_vi)               
                    
#                    gmdata_obs_vi = np.cumsum(gmdata_obs_vi)*dt
#                    gmdata_sim_vi = np.cumsum(gmdata_sim_vi)*dt                                       
                    
                    gmdata_obs_vi = rms(gmdata_obs_vi)
                    gmdata_sim_vi = rms(gmdata_sim_vi)            
#                    gmdata_sim_vi1 = rms(gmdata_sim_vi1)   
                                 
                    fft_obs=fftpack.fft(gmdata_obs_vi)
                    fft_sim=fftpack.fft(gmdata_sim_vi)
#                    fft_sim1=fftpack.fft(gmdata_sim_vi1)
                    
                    fft_obs=np.abs(fft_obs)
                    fft_sim=np.abs(fft_sim)
#                    fft_sim1=np.abs(fft_sim1)
                    
                    sample_freq = fftpack.fftfreq(gmdata_obs_vi.size, d=0.02)
                
                   
                    #Parameters for window    
                    df=1/dt
                    lt=num_pts
                    pad=5
                    #flexwin
                    e_s_c_name = str(ishot)+'.'+s0+'.win'

#                    filename = '/home/user/workspace/GMPlots/synth_VMs/Medium_VMs/flexwin/test_data_TRUE/MEASURE/ALL_WINs/'+e_s_c_name
#                    filename = '/home/user/workspace/GMPlots/synth_VMs/Medium_VMs/flexwin/test_data_TRUE/MEASURE/ALL_WINs_cc02/'+e_s_c_name                    
                    filename = '/home/user/workspace/GMPlots/synth_VMs/Medium_VMs/flexwin/test_data_TRUE/MEASURE/ALL_WINs_V2/'+e_s_c_name

                    t_on, t_off, td_shift, cc = read_flexwin(filename)
                    
                    if (t_on>0) and ((t_off-t_on)>0):
#                        if (t_off> R_Time_record[1,ishot-1,i]):
#                            t_off = R_Time_record[1,ishot-1,i]
#                            count_t_end_corrected = count_t_end_corrected+1
                            
                        tx_on=int(t_on/dt)
                        tx_off=int(t_off/dt)
                        #Window for isolated filter
#                        wd=winpad(lt,tx_off,tx_on,pad) 
                        wd=np.zeros(gmdata_obs_vi.shape)
                        wd[tx_on:tx_off+1]=1
                        
                        rwm = rwm_wd(gmdata_obs_vi,gmdata_sim_vi,num_pts, dt,wd);      
    #                    gmdata_obs_vi = 
                        
#                        ncc_R[i,k], td_max = ncc(gmdata_sim_vi,gmdata_obs_vi,num_pts, delta_T,dt)    
    
                        gmdata_sim_vi = np.multiply(gmdata_sim_vi,wd)
                        gmdata_obs_vi = np.multiply(gmdata_obs_vi,wd) 
                        
                        ncc_R[i,k], td_max_R[i,k] = ncc2(gmdata_sim_vi,gmdata_obs_vi,num_pts, delta_T,dt)                          
                        
#                        td_max_R[i,k] = TauP(gmdata_sim_vi,gmdata_obs_vi,num_pts, dt,wd)
                        
                        td_max_all[i,k,ishot-1] = td_max_R[i,k] 
                        tp_max_all[i,k,ishot-1] = TauP(gmdata_sim_vi,gmdata_obs_vi,num_pts, dt,wd)
                        index_all[i,k,ishot-1] = 1 
                        
#                        rwn_max_count.append(rwm)
#                        ncc_max_count.append(ncc_R[i,k])
#                        cc_max_count.append(cc)
#                        td_max_count.append(td_max_R[i,k])
#                        tp_max_count.append(tp_max_all[i,k,ishot-1])                       
#                        win_max_count.append(t_off-t_on)
#                        dist_max_count.append(distance)
                        
                        
                        g1 = float("{0:.2f}".format(rwm)); ncc1=float("{0:.2f}".format(ncc_R[i,k])); td1=float("{0:.2f}".format(td_max_R[i,k]));
                        
                        if(k==0):
                            plt.subplot(1,3,1)
                        #    plt.figure(figsize=(10,2.5))    
                            plt.plot(t,gmdata_obs_vi,c='k',linestyle='solid')
                            plt.plot(t,gmdata_sim_vi,c='r',linestyle='solid')
    #                        plt.plot(t,gmdata_sim_vi1,c='r',linestyle='solid')
                        
                            plt.title('Event ['+str(ishot)+']:'+sNames[ishot-1]+', '+statname, loc='center')

                            plt.ylabel(vxyz)
    
#                            plt.hlines(-4,R_Time_record[0,ishot-1,i], R_Time_record[1,ishot-1,i], colors='r', linestyles='solid')
                            plt.hlines(-3,tx_on*dt, tx_off*dt, colors='k', linestyles='solid')
                            ax = plt.gca()
                            ax.text(0.05,-0.5, 'rwm='+str(g1)+',TauP='+str(td1)+'s'+',ncc='+str(ncc1), horizontalalignment='left', verticalalignment='top', transform = ax.transAxes, fontsize=9.0)
                            if(rwm>5) or ((t_off-t_on)<20):
                                print('Removing '+statname+str(vxyz))                             
                            
                            ax.axis('on')      
                            rwm0=rwm
                            
                        if(k==1):
                            plt.subplot(1,3,2)
                        #    plt.figure(figsize=(10,2.5))    
                            plt.plot(t,gmdata_obs_vi,c='k',linestyle='solid')
                            plt.plot(t,gmdata_sim_vi,c='r',linestyle='solid')
    #                        plt.plot(t,gmdata_sim_vi1,c='r',linestyle='solid')
                        
                            plt.title('Event ['+str(ishot)+']:'+sNames[ishot-1]+', '+statname, loc='center')
    #                        plt.xlabel('Time (s)')
                            plt.ylabel(vxyz)
    
                            g1 = float("{0:.2f}".format(rwm)); ncc1=float("{0:.2f}".format(ncc_R[i,k])); td1=float("{0:.2f}".format(td_max_R[i,k]));
                            
#                            plt.hlines(-4,R_Time_record[0,ishot-1,i], R_Time_record[1,ishot-1,i], colors='r', linestyles='solid')
                            plt.hlines(-3,tx_on*dt, tx_off*dt, colors='k', linestyles='solid')                              
                      
                            ax = plt.gca()
                            ax.text(0.05,-0.5, 'rwm='+str(g1)+',TauP='+str(td1)+'s'+',ncc='+str(ncc1), horizontalalignment='left', verticalalignment='top', transform = ax.transAxes, fontsize=9.0)
                            ax.axis('on')
                            if(rwm>5) or ((t_off-t_on)<20):
                                print('Removing '+statname+str(vxyz))       
                            rwm1=rwm
                    
                        if(k==2):
                            plt.subplot(1,3,3)
                        #    plt.figure(figsize=(10,2.5))    
                            plt.plot(t,gmdata_obs_vi,c='k',linestyle='solid')
                            plt.plot(t,gmdata_sim_vi,c='r',linestyle='solid')
                            plt.title('Event ['+str(ishot)+']:'+sNames[ishot-1]+', '+statname, loc='center')
                            plt.xlabel('Time (s)')
                            plt.ylabel(vxyz)
                            ax = plt.gca()
                            ax.axis('on')
#                            plt.grid()            
                                                       
                            g1 = float("{0:.2f}".format(rwm)); ncc1=float("{0:.2f}".format(ncc_R[i,k])); td1=float("{0:.2f}".format(td_max_R[i,k]));
                            
#                            plt.hlines(-4,R_Time_record[0,ishot-1,i], R_Time_record[1,ishot-1,i], colors='r', linestyles='solid')
                            plt.hlines(-3,tx_on*dt, tx_off*dt, colors='k', linestyles='solid')                        
                            ax = plt.gca()
                            ax.text(0.05,-0.5, 'rwm='+str(g1)+',TauP='+str(td1)+'s'+',ncc='+str(ncc1), horizontalalignment='left', verticalalignment='top', transform = ax.transAxes, fontsize=9.0)
                            ax.axis('on')
                            if(rwm>5) or ((t_off-t_on)<20):
                                print('Removing '+statname+str(vxyz))        
                            rwm2=rwm
                        
#                        if (ncc_R[i,k]<0.5) or ((t_off-t_on)<20):
                        if(rwm>5) or ((t_off-t_on)<20) or (ncc_R[i,k]<0.5):
                            index_all[i,k,ishot-1]=0
                        else:
                            rwn_max_count.append(rwm)
                            ncc_max_count.append(ncc_R[i,k])
                            cc_max_count.append(cc)
                            td_max_count.append(td_max_R[i,k])
#                            td_max_count.append(td_shift)
                            tp_max_count.append(tp_max_all[i,k,ishot-1])                       
                            win_max_count.append(t_off-t_on)
                            dist_max_count.append(distance)                            

                    print('Event ['+str(ishot)+']:'+sNames[ishot-1]+', '+statname+vxyz)
                    print('rwm='+str(g1)+',TauP='+str(td1)+'s'+',ncc='+str(ncc1))
                    print('flexwin window ='+str(tx_on*dt)+'s to '+str(tx_off*dt)+'s')
                    print('Recorded time ='+str(R_Time_record[0,ishot-1,i])+'s to '+str(R_Time_record[1,ishot-1,i])+'s')

                plt.show()
#                    
#                if (R_ishot_new[i]==1):
#                    stat_count.append(statname)
#                    num_count.append(i)
                    
#        plt.figure(figsize=(10,2.5))       
#        plt.subplot(1,2,1)
#        plt.plot(ncc_R[:,0],c='k',linestyle='solid')
#        plt.xlabel('NCC '+GV[0])
#        plt.grid()
#        plt.subplot(1,2,2)
#        plt.plot(td_max_R[:,0],c='k',linestyle='solid')
#        plt.xlabel('Td_max')
#        plt.ylabel('Time [s]')
#        plt.show()        
#                
#        plt.figure(figsize=(10,2.5))       
#        plt.subplot(1,2,1)
#        plt.plot(ncc_R[:,1],c='k',linestyle='solid')
#        plt.xlabel('NCC'+GV[1])
#        plt.grid()
#        plt.subplot(1,2,2)
#        plt.plot(td_max_R[:,1],c='k',linestyle='solid')
#        plt.xlabel('Td_max')
#        plt.ylabel('Time [s]')
#        plt.show()        
#        
#        plt.figure(figsize=(10,2.5))       
#        plt.subplot(1,2,1)
#        plt.plot(ncc_R[:,2],c='k',linestyle='solid')
#        plt.xlabel('NCC'+GV[2])
#        plt.grid()
#        plt.subplot(1,2,2)
#        plt.plot(td_max_R[:,2],c='k',linestyle='solid')
#        plt.xlabel('Td_max')
#        plt.ylabel('Time [s]')
#        plt.show()        
        
#        print(num_count)
#        print(stat_count)
#        ncc_all[:,:,ishot-1] = ncc_R 
#        td_max_all[:,:,ishot-1] = td_max_R
#        all_stat=all_stat+len(num_count)
        #R_ishot_arr=R_ishot_new.reshape(-1)
#        if (ishot%5==0):    
#            print('finish source'+str(ishot))
        print('finish source'+str(ishot))
        print(' ')
    else:
        print('no record for source'+str(ishot))
        
#    np.savetxt('R_ishot_rwm/R_ishot_'+str(ishot)+'.txt', R_ishot_new)    
#print(all_stat)
#print(np.sum(ncc_all)/(3*all_stat))
#print(np.sum(td_max_all)/(3*all_stat))


#np.savetxt('td_max_count.txt',td_max_count)
#np.savetxt('tauP_count.txt',td_max_count)
#np.savetxt('ncc_max_count.txt',ncc_max_count) 
        
#np.savetxt('td_max_all.txt',td_max_all.reshape(-1))
#np.savetxt('index_all.txt',index_all.reshape(-1))
        
#np.savetxt('td_max_all_tshift_wp005.txt',td_max_all.reshape(-1))

#np.savetxt('tp_max_all_tshift.txt',tp_max_all.reshape(-1))
#np.savetxt('index_all_tshift.txt',index_all.reshape(-1))        
#np.savetxt('index_all_ncc_gt005.txt',index_all.reshape(-1))
#np.savetxt('index_all_ncc_gt000.txt',index_all.reshape(-1))    
#np.savetxt('index_all_ncc_gt002.txt',index_all.reshape(-1))       
        
np.savetxt('index_all_ncc_gt005_V2.txt',index_all.reshape(-1))        
        
#np.savetxt('index_all_ncc_gt005_t_end_corrected.txt',index_all.reshape(-1))      
#np.savetxt('index_all_ncc_gt005_tshift_20s_t_end_corrected.txt',index_all.reshape(-1))        
#np.savetxt('index_all_ncc_gt002_tshift_20s_t_end_corrected.txt',index_all.reshape(-1))        
#np.savetxt('index_all_ncc_gt005_tshift_20s_new.txt',index_all.reshape(-1))           
        
ct=0
ct_sum=0
for ishot in range(1,nShot+1):
    if (np.sum(index_all[:,:,ishot-1])<7):
        ct=ct+1
        ct_sum=ct_sum+np.sum(index_all[:,:,ishot-1])
        print(ishot)
print('ct')
print(ct)
print(ct_sum)

plt.figure(figsize=(5,5))       
plt.plot(dist_max_count,win_max_count,'ro')
plt.ylabel('Window length [s]')
plt.xlabel('Distance [km]')
#plt.xlim([0, 7])
plt.show()     

plt.figure(figsize=(5,5))       
plt.plot(dist_max_count,np.abs(td_max_count),'ro')
plt.ylabel('Abs(TauP) [s]')
plt.xlabel('Distance [km]')
#plt.xlim([0, 7])
plt.show()   

plt.figure(figsize=(5,5))       
plt.plot(dist_max_count,td_max_count,'ro')
plt.ylabel('TauP [s]')
plt.xlabel('Distance [km]')
#plt.xlim([0, 7])
plt.show()       
     
plt.figure(figsize=(5,5))       
plt.plot(rwn_max_count,ncc_max_count,'ro')
plt.xlabel('rwn')
plt.ylabel('ncc')
plt.xlim([0, 8])
#plt.ylim([0.5, 1])
plt.show()        
#        
#
plt.figure(figsize=(5,5))       
plt.plot(rwn_max_count,cc_max_count,'ro')
plt.xlabel('rwn')
plt.ylabel('cc')
plt.xlim([0, 8])
plt.show()    


#sys.stdout = orig_stdout
#f.close()
#plt.figure(figsize=(5,5))       
#plt.plot(np.abs(td_max_count),win_max_count,'ro')
#plt.ylabel('Window length [s]')
#plt.xlabel('Abs(TauP) [s]')
#plt.xlim([0, 7])
#plt.show()        
#
#hist1 = pd.Series(win_max_count)
#win_short = 40
#hist1.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
##plt.vlines(win_short,0, 50, colors='k', linestyles='dashed')
#plt.xlabel('Window length [s]',fontsize=14)
#
#plt.figure(figsize=(5,5)) 
#hist2 = pd.Series(td_max_count)
#win_mean = np.median(td_max_count)
#hist2.plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
##plt.vlines(win_mean,0, 50, colors='k', linestyles='dashed')
#plt.xlabel('Td_max [s]',fontsize=14)
#plt.xlim([-9, 9])
#plt.ylim([0, 300])
#
#hist3 = pd.Series(tp_max_count)
#win_mean = np.mean(tp_max_count)
#hist3.plot.hist(grid=True, bins=100, rwidth=0.9, color='#607c8e')
#plt.vlines(win_mean,0, 50, colors='k', linestyles='dashed')
#plt.xlabel('TauP [s]',fontsize=14)
#plt.ylabel('Waveform count',fontsize=14)
#plt.xlim([-10, 10])

#rwn_max_count0 = rwn_max_count
#rwn_max_count=np.loadtxt('rwm_init.txt')
#rwn_max_count0=np.loadtxt('rwm_inv.txt')
#rwn_max_count[618]=0;rwn_max_count[619]=0;rwn_max_count[2489]=0;
#rwn_max_count0[618]=0;rwn_max_count0[619]=0;rwn_max_count0[2489]=0;
#rwn_max_count_new=[]
#rwn_max_count0_new=[]

#for i_rwm in range(0, len(rwn_max_count)):
#    if (rwn_max_count[i]>5):
#        rwn_max_count[i]=0
#        rwn_max_count0[i]
##        rwn_max_count_new.append(rwn_max_count[i])
##        rwn_max_count0_new.append(rwn_max_count0[i])        
        
#plt.figure(figsize=(5,5)) 
#hist4 = pd.Series(rwn_max_count)
#hist4.plot.hist(grid=True, bins=2000, rwidth=0.9, color='#607c8e')
##hist4.plot.hist(grid=True, bins=100, rwidth=0.9, color='r')
##plt.vlines(np.median(hist4),0, 1000, colors='k', linestyles='dashed')
#
##hist40 = pd.Series(rwn_max_count-rwn_max_count0)
##hist40.plot.hist(grid=True, bins=10, rwidth=0.9, color='r',alpha = 0.5)
###hist4.plot(edgecolor='k', linewidth=1.2)
###plt.vlines(np.median(hist40),0, 1000, colors='r', linestyles='dashed')
#plt.xlim([0, 5])
#plt.xlabel('Relative waveform misfit',fontsize=14)
#plt.ylabel('Frequency',fontsize=14)
##plt.xlabel('RWM improvement',fontsize=14)
###np.savetxt('rwm_init.txt',hist4)
###np.savetxt('rwm_inv.txt',hist4)
###np.savetxt('rwm_inv_ncc002.txt',hist40)
##np.savetxt('rwm_init_ncc002_new.txt',hist4)
#np.savetxt('rwm_inv_ncc002_new.txt',hist4)
#
#plt.figure(figsize=(5,5)) 
#hist5 = pd.Series(td_max_count)
##win_short = 40
#hist5.plot.hist(grid=True, bins=10, rwidth=0.9, color='#607c8e')
#plt.vlines(np.median(td_max_count),0, 50, colors='k', linestyles='dashed')
#plt.xlim([10, 10])
#plt.xlabel('Td__max',fontsize=14)

#np.savetxt('Td_max_init.txt',hist5)