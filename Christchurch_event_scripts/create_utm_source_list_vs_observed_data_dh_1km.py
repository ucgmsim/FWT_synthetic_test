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

def read_srf(srf_file):
    """
    Function for reading srf files
    """
    with open(srf_file, 'r') as f:
        lines = f.readlines()
        line5=lines[5].split()
        Si=np.zeros((1,3))
        Si[0,0]=line5[0]
        Si[0,1]=line5[1]
        Si[0,2]=line5[2]
    
    return Si

def read_list_srf(list_file):

    with open(list_file, 'r') as f:
        lines = f.readlines()
    nSource = len(lines) 
    sNames = [] 
    for i in range(0,nSource,1):
        line_i=lines[i].split('.')
        sNames.append(line_i[0])
        
    return nSource, sNames

def read_list_event(list_file):

    with open(list_file, 'r') as f:
        lines = f.readlines()
    nSource = len(lines) 
    sNames = [] 
    for i in range(0,nSource,1):
        line_i=lines[i][0:-1]
        sNames.append(line_i)
        
    return nSource, sNames

def ll2xy_conv(MODEL_LON,MODEL_LAT,MODEL_ROT,ELON,ELAT,EDEP,NX,NY,HH):
       
    XAZIM = MODEL_ROT+90.0
    
    cmd = 'echo '+ str(ELON) + " " + str(ELAT) +'| ./ll2xy mlon='+str(MODEL_LON)+' mlat='+str(MODEL_LAT)+' xazim='+str(XAZIM)
    stdout = Popen(cmd, shell=True, stdout=PIPE).stdout
    output = stdout.read()    
    EXY = output.split()
   # input('-->')

    XSRC = int(0.5*NX + float(EXY[0])/HH)

    YSRC = int(0.5*NY + float(EXY[1])/HH)

    ZSRC = int(EDEP/HH + 0.5) + 1 
    
    print([XSRC, YSRC, ZSRC])   
    
    return XSRC, YSRC, ZSRC

def write_srf_source(S,sNames,S_LL):
    filename1='SOURCE_SRF_corrected_dh_1km_RGT.txt' 
    Ns=len(S)
    print(filename1)
    fid = open(filename1,'w')
    fid.write("%4d\n" %(Ns))    
    count=0
    for i in range(0,Ns,1):
        fid.write("%4d%4d%4d %20s %20f%20f%10f\n" %(S[count,0],S[count,1],S[count,2],sNames[count],S_LL[count,0],S_LL[count,1],S_LL[count,2]))        
        count=count+1    
    
    
def read_source(source_file):
    
    with open(source_file, 'r') as f:
        lines = f.readlines()    
    line0=lines[0].split()
    nShot=int(line0[0])
    S=np.zeros((nShot,3))
    for i in range(1,nShot+1):
        line_i=lines[i].split()
        S[i-1,0]=line_i[0]
        S[i-1,1]=line_i[1]
        S[i-1,2]=line_i[2]
    
    return nShot, S
#########################################
nSource, sNames = read_list_srf('list_srf.txt')

nSource_obs, sNames_obs = read_list_event('list_148events.txt')
S = np.zeros((nSource_obs,3))
S_LL = np.zeros((nSource_obs,3))

sNames_new =[]

NX = 130
NY = 130
NZ = 60
HH = 1.0

#MODEL_LON = 172.29999979889402
#MODEL_LAT = -43.59999583292219
#MODEL_ROT = 0.0
MODEL_LON = 172.299999799
MODEL_LAT = -43.5999958329
MODEL_ROT = 0

i_obs = 0
for i in range(0,nSource,1):
    sName_split = sNames[i].split('m')
    if (sName_split[0] in sNames_obs):
        sNames_new.append(sNames[i])
        srf_file = 'Srf/'+str(sNames[i])+'.srf'
        print(srf_file)
        #input('-->')
        Si = read_srf(srf_file)
        ELON = Si[0,0]
        ELAT = Si[0,1]
        EDEP = Si[0,2]
        
        S_LL[i_obs,:] = [ELON, ELAT, EDEP]
    
        S[i_obs,:] = ll2xy_conv(MODEL_LON,MODEL_LAT,MODEL_ROT,ELON,ELAT,EDEP,NX,NY,HH)
               
#        dir_ob='Kernels/Vel_ob/Vel_ob_'+str(i_obs+1)
        #dir_ob='Vel_ob/Vel_ob_'+str(i_obs+1)
        #mkdir_p(dir_ob)
    
#        os.system('cp ~/ObservedGroundMotions/'+sName_split[0]+'/Vol1/data/velLF/* Kernels/Vel_ob/Vel_ob_'+str(i_obs+1))
#        os.system('cp ObservedGroundMotions/'+sName_split[0]+'/Vol1/data/velLF/* Vel_ob/Vel_ob_'+str(i_obs+1))        
        i_obs = i_obs+1        
    
write_srf_source(S,sNames_new,S_LL)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
