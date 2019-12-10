#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 14:19:45 2018

@author: user
"""

import numpy as np
#import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift
#from scipy import signal
from scipy.signal import butter, lfilter, hilbert, find_peaks
from scipy import signal
from scipy.optimize import curve_fit
from qcore import timeseries
import os
import scipy

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

######################################
def normpdf_python(x, mu, sigma):
   return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1*(x-mu)**2/2*sigma**2)

#################################################
def Eqn(x,b0,b1,b2,b3,b4):
    """
    Nonlinear regression for fitting the Gaussian wavelet
    """       
    return b0*np.exp(-0.5*b1**2*(x-b2)**2)*np.cos(b3*(x-b4))

def envelope(signal2):
    """
    Envelope the Gaussian wavelet
    """      
    analytic_signal = hilbert(signal2)
    amplitude_envelope = np.abs(analytic_signal)  
    return amplitude_envelope
    
def nextpow2(x):
    """returns the smallest power of two that is greater than or equal to the
    absolute value of x.
    """
    res = np.ceil(np.log2(x))
    return res.astype('int') #we want integer values only but ceil gives float        
    
def computebandfftfilter_gauss(signal0,dt,fc,sigma0,lTime):
    """
    Compute a narrow-band filter in frequency
    """    
    signal1=np.concatenate((signal0[0]*np.linspace(1,1,100), signal0, signal0[-1]*np.linspace(1,1,100)))
    fc=fc*2*np.pi
    sigma=sigma0*2*np.pi

    NFFT = 2**nextpow2(len(signal1))
    fourier=scipy.fft(signal1,NFFT)    
    
#    %dt=.5;
    f   = 2*np.pi*(1/dt)*(np.linspace(0,NFFT-1,NFFT))/NFFT #CORRECT! 
    fil = np.exp((-(f-fc)**2)/(2*sigma**2))
    signfiltf = np.multiply(fil,fourier)

    #new ifft 
    points = int(NFFT/2)
    fas = signfiltf[range(points)]
    
    # Using only first half, rebuild full FAS by taking conjugate
    fas_eqsig = np.zeros(len(signfiltf), dtype=complex)
    #! Do not start from fas[0]
    fas_eqsig[1:NFFT // 2] = fas[1:]
    fas_eqsig[NFFT // 2 + 1:] = np.flip(np.conj(fas[1:]), axis=0) 
    # Inverse the rebuilt FAS to obtain the time series
    sft = np.fft.ifft(fas_eqsig, n=NFFT)    
    
#    %signal=interp1(0:length(sft)-1,sft,time);
    stfinal=sft[100:len(signal1)-100]      
    
    return stfinal

def source_adj_gsdf(gmdata_sim,gmdata_obs,IsolationFilter,num_pts,dt):
    """
    Calculated the adjoint source using S-arrival window and gsdf measurement for delay time 
    """        
    t = np.arange(num_pts)*dt
    ts=np.flip(-t[1:], axis=0)
    lTime = np.concatenate((ts,t), axis=0)#Lag time    
    
    #convolve the waveforms for the cross- and auto-correlagrams     
    cross = np.correlate(IsolationFilter,gmdata_obs,'full')
    auto = np.correlate(IsolationFilter,gmdata_sim,'full')  
    
    #GSDF Parameters 
    w0=2*np.pi/(lTime[-1])             
#    wN=2*np.pi/(2*dt)
#    w(:,1)=-wN:w0:wN
    wf=w0*np.linspace(-int(num_pts/2),int(num_pts/2),num_pts)  
    #fi = [0.05, 0.075, 0.1]

    fi = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
#    fi = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
#    Tw = 2/np.mean(fi)      # Effective window
#    sw = 2*np.pi*0.72/Tw;      # Sigma w ~ 0.2827433388230814
    sw=0.1
#    sw=0.05    
    
#    #% A local maximum will be selected closest to 0-lag
#    I_O=np.argmax(cross)
#    I_S=np.argmax(auto)     

    I_O, peaks_O = find_peaks(np.abs(hilbert(cross))/np.max(np.abs(hilbert(cross))), height=0.25)
    I_S, peaks_S = find_peaks(np.abs(hilbert(auto))/np.max(np.abs(hilbert(auto))), height=0.25)

    PkO = peaks_O.get("peak_heights", "")
    PkS = peaks_S.get("peak_heights", "")

    if (I_O==[] or I_S==[]):
        I_O=np.argmax(cross)
        I_S=np.argmax(auto)
    else:
        I_O_min = np.argmin(np.multiply((1+np.abs(lTime[I_O]))**2,np.abs(1-PkO)))
        I_O = I_O[I_O_min]

        I_S_min = np.argmin(np.multiply((1+np.abs(lTime[I_S]))**2,np.abs(1-PkS)))
        I_S = I_S[I_S_min]
    
    ##Windowing
    win1=np.exp(-(0.5*sw**2)*(lTime-lTime[I_O])**2)
    win2=np.exp(-(0.5*sw**2)*(lTime-lTime[I_S])**2)   
    
    #
    WO = np.multiply(win1,cross)
    WS = np.multiply(win2,auto)
    WS = WS*np.max(WO)/np.max(WS) #Normalized window by amplitude
    #% Parameters for "bootstraping"
    InOR=np.argmax(WO)
    InSR=np.argmax(WS)     
       
    #% Isolation filter FFT for perturbation kernel
    tff=np.conj(fftshift(fft(IsolationFilter)))*1/num_pts   
    
    adj_sim_decompose = np.zeros((len(fi),num_pts))
    adj_sim_sum = np.zeros(num_pts)
    TauP_arr = np.zeros(len(fi))    
    
    ne = int(np.min([2/np.min(fi)/dt,num_pts/2]))    #% Effective bandwidth for inversion
    
    for i in range(0,len(fi)):  
        si = 0.1*fi[i]
        #Crosscorrelagram and Autocorrelagram filtering
        dO=computebandfftfilter_gauss(WO,dt,fi[i],si,lTime);
        dS=computebandfftfilter_gauss(WS,dt,fi[i],si,lTime);    
               
        #    % Check bootstraping
        InO=np.argmax(np.real(dO))
        InS=np.argmax(np.real(dS))   
        
        BS = 1; Cn = 0;
        while BS == 1 or Cn < 10:
            InO=int(InO)
            if (lTime[InO] < lTime[InOR]+0.51/fi[i]) and (lTime[InO] >= lTime[InOR]-0.51/fi[i]):
                BS = 0
            elif (lTime[InO] >= (lTime[InOR]+0.45/fi[i])):
                InO=InO-np.round(1/fi[i]/dt)
            elif (lTime[InO] < lTime[InOR]-0.45/fi[i]):
                InO=InO+np.round(1/fi[i]/dt)
            Cn = Cn+1
            
        BS = 1; Cn = 0;
        while BS == 1 or Cn < 10:
            InS=int(InS)            
            if (lTime[InS] < lTime[InSR]+0.51/fi[i]) and (lTime[InS] >= lTime[InSR]-0.51/fi[i]):
                BS = 0
            elif (lTime[InS] >= (lTime[InSR]+0.45/fi[i])):
                InS=InS-np.round(1/fi[i]/dt)
            elif (lTime[InS] < lTime[InSR]-0.45/fi[i]):
                InS=InS+np.round(1/fi[i]/dt)
            Cn = Cn+1  

        # Five parameter Gaussian wavelet fitting    
        Ao = np.max(envelope(np.real(dO))); Io = np.argmax(envelope(np.real(dO)));
        As = np.max(envelope(np.real(dS))); Is = np.argmax(envelope(np.real(dS))); 
        ##Constrain the initial values   
        # Parameters for curve_fit
        wi=2*np.pi*fi[i]  
        
        try:
            GaO, params_covariance = curve_fit(Eqn, lTime[Io-ne-1:Io+ne], np.real(dO[Io-ne-1:Io+ne]))
            GaS, params_covariance = curve_fit(Eqn, lTime[Is-ne-1:Is+ne], np.real(dS[Is-ne-1:Is+ne]))     
        except:
            GaO = [Ao, 2*np.pi*si, lTime[Io], 2*np.pi*fi[i], lTime[InO]]
            GaS = [As, 2*np.pi*si, lTime[Is], 2*np.pi*fi[i], lTime[InS]]   

#        GaO, params_covariance = curve_fit(Eqn, lTime[Io-ne-1:Io+ne], np.real(dO[Io-ne-1:Io+ne]),bounds=(0,[Ao, 2*np.pi*si, lTime[Io], 2*np.pi*fi[i], lTime[InO]]))
#        GaS, params_covariance = curve_fit(Eqn, lTime[Is-ne-1:Is+ne], np.real(dS[Is-ne-1:Is+ne]),bounds=(0,[As, 2*np.pi*si, lTime[Is], 2*np.pi*fi[i], lTime[InS]]))    
        
#        % Check fitting
        if ((GaO[0]/GaS[0]) > 10**5) or np.abs(GaO[4]-GaS[4]) > lTime[-1]/2:
            GaO = [Ao, 2*np.pi*si, lTime[Io], 2*np.pi*fi[i], lTime[InO]]
            GaS = [As, 2*np.pi*si, lTime[Is], 2*np.pi*fi[i], lTime[InS]]      
         
        wP=((si**2)*wf+(sw**2)*wi)/(sw**2+si**2)
        wPP=((si**2)*wf-(sw**2)*wi)/(sw**2+si**2)
        siP=((si**2)*(sw**2)/(sw**2+si**2))**0.5    
        #Estimate waveform perturbation kernel (WPK)
        IW=(siP/(sw*GaS[0]))*np.multiply(np.exp(-0.5*(wf-2*np.pi*fi[i])**2/(sw**2+si**2)),np.divide(tff,wP))+\
        (siP/(sw*GaS[0]))*np.exp(-0.5*(wf+2*np.pi*fi[i])**2/(sw**2+si**2))*tff/wPP
        
        IW[0:int(len(IW)/2)]=0*IW[0:int(len(IW)/2)]
        
        itff = ifft(fftshift(num_pts*IW))  
    
        #Save the GSDF measurements
        TauP_arr[i] = GaO[4]-GaS[4]; #% delta_P
        
#        Jp = np.real(itff)
        Jp = np.imag(itff)        
#        Jp = -np.imag(itff)
        adj_sim_decompose[i,:] = np.flip(Jp,axis=0)*TauP_arr[i]   
        
        adj_sim_sum = adj_sim_sum + adj_sim_decompose[i,:]     
            
    return adj_sim_sum, TauP_arr

def write_adj_source_ts(s1,v1,mainfolder,mainfolder_source,source,dt):
    """
    write adjoint source
    """   
    vs1=v1.split('.')
    timeseries.seis2txt(source,dt,mainfolder_source,vs1[0],vs1[1])
    return	   
##############################################
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
    
def rms(stat_data):
    
    num_pts=len(stat_data)    
    D = (np.sum(np.square(stat_data))/num_pts)**0.5
   
    return stat_data/D      

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
    if(t_on<10):
        t_on=10
    if(t_off>lt-10):
        t_off=lt-10

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

def time_shift_emod3d(data,delay_Time,dt):
    n_pts = len(data)
    ndelay_Time = int(delay_Time/(dt))
    data_shift = np.zeros(data.shape)
    data_shift[0:n_pts-ndelay_Time] = data[ndelay_Time:n_pts]
    return data_shift
###################################################
#statnames = ['A1' ,'A2' ,'A3' ,'A4' ,'A5' ,'A6' , 'A7', 'B1' ,'B2' ,'B3' ,'B4' ,'B5' ,'B6' , 'B7','C1' ,'C2' ,'C3' ,'C4' ,'C5' ,'C6' ,'C7','D1' ,'D2' ,'D3' ,'D4' ,'D5' ,'D6' ,'D7','E1' ,'E2' ,'E3' ,'E4' ,'E5' ,'E6' ,'E7','F1' ,'F2' ,'F3' ,'F4' ,'F5' ,'F6','F7','G1' ,'G2' ,'G3' ,'G4' ,'G5' ,'G6','G7']
station_file = '../../../StatInfo/STATION.txt'    
nRec, R, statnames = read_stat_name(station_file)
#statnames=statnames[0:10]
print('statnames')
print(statnames)

GV=['.090','.000','.ver']
GV_ascii=['.x','.y','.z']

mainfolder='../../Vel_es/Vel_es_i/'
mainfolder_o='../../Vel_ob_new_V2/Vel_ob_i/'
mainfolder_source='../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/'
os.system('rm ../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/*.*')

print(mainfolder_o)
_, num_pts, dt, shift  = readGP_2('../../Vel_ob_new_V2/Vel_ob_i','CBGS.000')
num_pts=int(num_pts)
t = np.arange(num_pts)*dt
############/nesi/nobackup/nesi00213/RunFolder/tdn27/rgraves/Adjoint/Syn_VMs/Kernels/#########################
fs = 1/dt
lowcut = 0.05
#highcut = 0.05
highcut = 0.1
#ndelay_T=int((3/0.1)/(dt))
delta_T=5
flo=0.1
delay_Time=(3/flo)

fc = highcut  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(4, w, 'low')

source_file='../../../StatInfo/SOURCE.txt'

nShot, S = read_source(source_file)
wr_arr = np.loadtxt('../../../../Kernels/Iters/iter1/Dump/geo_correlation.txt')
wr=np.reshape(wr_arr,[nRec,nShot])

################################
fi1=open('iShot.dat','r')
ishot=int(np.fromfile(fi1,dtype='int64'))
fi1.close()
print('ishot='+str(ishot))

#R_ishot_arr=np.loadtxt('../../../../Kernels/Iters/iter1/Dump/R_ishot_'+str(ishot)+'.txt')
#R_all_arr=np.loadtxt('../../../../Kernels/index_all_ncc_gt005.txt')
#R_all_arr=np.loadtxt('../../../../Kernels/index_all_ncc_gt002.txt')
#R_all_arr=np.loadtxt('../../../../Kernels/index_all_ncc_gt005_V2_10s.txt')
R_all_arr=np.loadtxt('../../../../Kernels/index_all_ncc_pyflex_gt05_V2_excluded.txt')
R_all=R_all_arr.reshape([nRec,3,nShot])

for i,statname in enumerate(statnames):

    distance=((R[i,1]-S[ishot-1,1])**2+(R[i,2]-S[ishot-1,2])**2+(R[i,0]-S[ishot-1,0])**2)**(0.5)

    for k in range(0,3):
        
        source_adj=np.zeros(num_pts)
        
        s0=statname+GV[k]
        v0=statname+GV_ascii[k]
        
        if((distance<200) and (distance>0) and (R_all[i,k,ishot-1]==1)):    
            print('ireceiver='+str(i))
#            wr_ij = wr[nRec*(ishot-1)+i]
#           wr_ij = wr[nShot*(i)+ishot-1]
            wr_ij = wr[i,ishot-1]
   
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
    
            #stat_data_0_O  = np.multiply(signal.tukey(int(num_pts),1.0),stat_data_0_O)
        
#            stat_data_0_S = rms(stat_data_0_S)*wr_ij    
#            stat_data_0_O = rms(stat_data_0_O)*wr_ij
            stat_data_0_S = (stat_data_0_S)*wr_ij    
            stat_data_0_O = (stat_data_0_O)*wr_ij
            
            #Parameters for window    
            df=1/dt
            lt=num_pts
            pad=5
            #flexwin
            e_s_c_name = str(ishot)+'.'+s0+'.win'
#            filename = '../../../../Kernels/ALL_WINs_V2/'+e_s_c_name
            filename = '../../../../Kernels/ALL_WINs_pyflex_temp/'+e_s_c_name

            t_on, t_off, td_shift, cc = read_flexwin(filename)
            tx_on=int(t_on/dt)
            tx_off=int(t_off/dt)
            
            #Window for isolated filter
            wd=winpad(lt,tx_off,tx_on,pad)  
            
            stat_data_0_S = np.multiply(stat_data_0_S,wd)
            stat_data_0_O = np.multiply(stat_data_0_O,wd)             
            #Isolation filters  
            IsolationFilter_0 = np.multiply(stat_data_0_S,wd)    
    
            source_adj,_ = source_adj_gsdf(stat_data_0_S,stat_data_0_O,IsolationFilter_0,num_pts,dt)
#            source_adj = signal.detrend(source_adj)
        
        write_adj_source_ts(s0,v0,mainfolder,mainfolder_source,source_adj,dt)




    

    
    
    
