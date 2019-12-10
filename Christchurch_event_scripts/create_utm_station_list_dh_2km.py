#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:56:20 2019

@author: user
"""
#import os
from subprocess import Popen, PIPE
import numpy as np
import os
#import matplotlib.pyplot as plt

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

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

def read_list_station_ll(list_file):

    with open(list_file, 'r') as f:
        lines = f.readlines()
    nStat = len(lines) 
    stat_names = [] 
    R_list=np.zeros((nStat,2))
    
    for i in range(0,nStat,1):
        line_i=lines[i].split()

        R_list[i,0]=float(line_i[0])
        R_list[i,1]=float(line_i[1])
        stat_names.append(line_i[2])        
        
    return nStat, R_list, stat_names

#def ll2xy_conv(MODEL_LON,MODEL_LAT,MODEL_ROT,ELON,ELAT,EDEP,NX,NY,HH):
#       
#    XAZIM = MODEL_ROT+90.0
#    
#    cmd = 'echo '+ str(ELON) + " " + str(ELAT) +'| ./ll2xy mlon='+str(MODEL_LON)+' mlat='+str(MODEL_LAT)+' xazim='+str(XAZIM)
#    stdout = Popen(cmd, shell=True, stdout=PIPE).stdout
#    output = stdout.read()    
#    EXY = output.split()
#   # input('-->')
#
#    XSRC = int(0.5*NX + float(EXY[0])/HH)
#
#    YSRC = int(0.5*NY + float(EXY[1])/HH)
#
#    ZSRC = int(EDEP/HH + 0.5) + 1 
#    
#    print([XSRC, YSRC, ZSRC])   
#    
#    return XSRC, YSRC, ZSRC

def write_utm_station(R,statnames,R_LL):
    filename1='STATION_utm_dh_2km.txt' 
    Nr=len(R)
    print(filename1)
    fid = open(filename1,'w')
    fid.write("%4d\n" %(Nr))    
    count=0
    for i in range(0,Nr,1):
        fid.write("%4d%4d %20s %20f%20f\n" %(R[count,0],R[count,1],statnames[count],R_LL[count,0],R_LL[count,1]))        
        count=count+1    
    
    return    
    
#########################################
list_file = 'geoNet_stats+2019-04-18.ll'
nStat, R_list, stat_names = read_list_station_ll(list_file)

station_file = '/home/user/workspace/GMPlots/Christchurch_Events/INV1_148events_TRUE_DH_2km/Kernels/STATION.txt'  
nRec, R, statnames = read_stat_name(station_file)

#R = np.zeros((nRec,2))
R_LL = np.zeros((nRec,2))

for i,statname in enumerate(statnames):
 
      x = stat_names.index(statname)
      R_LL[i,0] = R_list[x,0]
      R_LL[i,1] = R_list[x,1]
    
write_utm_station(R,statnames,R_LL)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
