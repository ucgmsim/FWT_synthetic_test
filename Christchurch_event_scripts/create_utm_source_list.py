#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:56:20 2019

@author: user
"""
#import os
from subprocess import Popen, PIPE
import numpy as np
#import matplotlib.pyplot as plt

def read_srf(srf_file):
    """
    Function for reading srf files
    """
    with open(srf_file, 'r') as f:
        lines = f.readlines()
        line5=lines[5].split()
        Si=np.array(3)
        Si[0]=line5[0]
        Si[1]=line5[1]
        Si[2]=line5[2]
    
    return Si

def read_list_srf(list_file):

    with open(list_file, 'r') as f:
        lines = f.readlines()
    nSource = len(lines)-1 
    sNames = [] 
    for i in range(0,nSource,1):
        line_i=lines[i].split('.')
        sNames.append(line_i[0])
        
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
    filename1='SOURCE_SRF.txt' 
    Ns=len(S)
    print(filename1)
    fid = open(filename1,'w')
    fid.write("%4d\n" %(Ns))    
    count=0
    fid.write("%4d%4d%4d %20s %6f%6f%6f\n" %(S[count,0],S[count,1],S[count,2],sNames[count],S_LL[count,0],S_LL[count,1],S_LL[count,2]))        
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
S = np.zeros((nSource,3))
S_LL = np.zeros((nSource,3))

NX = 267
NY = 269
NZ = 75
HH = 0.400

MODEL_LON = 172.92310
MODEL_LAT = -43.28590
MODEL_ROT = 0.0


for i in range(0,nSource,1):
    srf_file = 'Srf/'+str(sNames[i])+'.srf'
    Si = read_srf(srf_file)
    ELON = Si[0]
    ELAT = Si[1]
    EDEP = Si[2]
    
    S_LL[i,:] = [ELON, ELAT, EDEP]

    S[i,:] = ll2xy_conv(MODEL_LON,MODEL_LAT,MODEL_ROT,ELON,ELAT,EDEP,NX,NY,HH)
    
write_srf_source(S,sNames,S_LL)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    