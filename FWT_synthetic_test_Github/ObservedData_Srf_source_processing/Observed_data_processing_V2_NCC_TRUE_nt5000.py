#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:36:09 2019

@author: user
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import resample
from qcore import timeseries

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

def read_e_stat_name(e_station_file):
    """
    Read the station list for an event
    """
    with open(e_station_file, 'r') as f:
        lines = f.readlines()
    nRec1=len(lines) 
    statnames1 = [] 
    for i in range(0,nRec1,1):
        line_i=lines[i].split('.')
        statnames1.append(line_i[0])
    return nRec1, statnames1

#def defy_record_matrix(station_file, e_station_file):
#    nRec1, statnames1 = read_e_stat_name(e_station_file)
# 
#    with open(station_file, 'r') as f:
#        lines = f.readlines()
#    line0=lines[0].split()
#    nRec=int(line0[0])
#    
#    R_record=np.zeros(nRec)
#    
#    nRec2=0
#    for i in range(1,nRec+1):
#        line_i=lines[i].split()
#        if(line_i[3] in statnames1):
##            fid.write("%s"%(lines[i]))      
#            nRec2=nRec2+1
#            R_record[i]=1
#
#    return nRec2, R_record
    
def defy_record_matrix(nRec, statnames, e_station_file):
    """
    Define the source-station matrix
    """    
    nRec1, statnames1 = read_e_stat_name(e_station_file)
   
    R_record=np.zeros(nRec)
    nRec2=0
    for i,statname in enumerate(statnames):
        if(statname in statnames1):   
            nRec2=nRec2+1
            R_record[i]=1

    return nRec2, R_record

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

##########################
station_file = 'STATION.txt'  
nRec, R, statnames = read_stat_name(station_file)

source_file='SOURCE.txt'
nShot, S = read_source(source_file)

R_all=np.zeros((nShot,nRec))

#statnames=statnames[0:4]
print('statnames')
print(statnames)

num_pts=5000; dt=0.02; N_sample=4;
GV=['.090','.000','.ver']

for ishot in range(1,nShot+1):
    
    os.system('ls  Vel_ob_org/Vel_ob_'+str(ishot)+'/*.ver | xargs -n 1 basename > tmp.txt')    
    e_station_file = 'tmp.txt'    
    _,  R_all[ishot-1,:] = defy_record_matrix(nRec, statnames, e_station_file)
    
    mainfolder_o='Vel_ob_org/Vel_ob_'+str(ishot)
    mainfolder_o_new='Vel_ob_nt5000/Vel_ob_'+str(ishot)+'/'
    os.system('rm Vel_ob_nt5000/Vel_ob_'+str(ishot)+'/*')
    
    for i,statname in enumerate(statnames):
        if (R_all[ishot-1,i]==1):
            
            s0=statname+GV[0]
            s1=statname+GV[1]
            s2=statname+GV[2]
            
            stat_data_0_shift = np.zeros(num_pts);stat_data_1_shift = np.zeros(num_pts);stat_data_2_shift = np.zeros(num_pts);
            
            _, num_pts0, dt0, shift  = readGP_2(mainfolder_o,s0)
            n_shift = -int(shift/dt)
            
            stat_data_0_O  = timeseries.read_ascii(mainfolder_o+'/'+s0)
            stat_data_1_O  = timeseries.read_ascii(mainfolder_o+'/'+s1)
            stat_data_2_O  = timeseries.read_ascii(mainfolder_o+'/'+s2)
            
            stat_data_0_O = resample(stat_data_0_O,int(num_pts0/N_sample))
            stat_data_1_O = resample(stat_data_1_O,int(num_pts0/N_sample))
            stat_data_2_O = resample(stat_data_2_O,int(num_pts0/N_sample))
            
            if n_shift>0:
                if (int(num_pts0/N_sample)<num_pts-n_shift):
                    stat_data_0_shift[n_shift:n_shift+int(num_pts0/N_sample)] = stat_data_0_O
                    stat_data_1_shift[n_shift:n_shift+int(num_pts0/N_sample)] = stat_data_1_O
                    stat_data_2_shift[n_shift:n_shift+int(num_pts0/N_sample)] = stat_data_2_O                            
                else:
                    stat_data_0_shift[n_shift:num_pts-100] = stat_data_0_O[0:num_pts-100-n_shift]
                    stat_data_1_shift[n_shift:num_pts-100] = stat_data_1_O[0:num_pts-100-n_shift]
                    stat_data_2_shift[n_shift:num_pts-100] = stat_data_2_O[0:num_pts-100-n_shift]     
            else:
                stat_data_0_shift[0:n_shift+int(num_pts0/N_sample)] = stat_data_0_O[-n_shift:int(num_pts0/N_sample)]
                stat_data_1_shift[0:n_shift+int(num_pts0/N_sample)] = stat_data_1_O[-n_shift:int(num_pts0/N_sample)]
                stat_data_2_shift[0:n_shift+int(num_pts0/N_sample)] = stat_data_2_O[-n_shift:int(num_pts0/N_sample)]                   
                                                       
    
            timeseries.seis2txt(stat_data_0_shift,dt,mainfolder_o_new,statname,'090')
            timeseries.seis2txt(stat_data_1_shift,dt,mainfolder_o_new,statname,'000')
            timeseries.seis2txt(stat_data_2_shift,dt,mainfolder_o_new,statname,'ver')  
            
R_arr=R_all.reshape(-1)
np.savetxt('R_all.txt', R_arr)
            
            