#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 14:19:45 2018

@author: user
"""

import numpy as np
#import matplotlib.pyplot as plt

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
   
###################################################
source_file='../../../StatInfo/SOURCE.txt'
station_file='../../../StatInfo/STATION.txt'

nShot, S = read_source(source_file)
nRec, R = read_source(station_file)
    
############/nesi/nobackup/nesi00213/RunFolder/tdn27/rgraves/Adjoint/Syn_VMs/Kernels/#########################

#statnames = ['AAA','BBB','CCC','DDD','EEE','FFF','GGG','HHH','III','JJJ','KKK','LLL','MMM','NNN','PPP','QQQ','RRR','SSS','TTT','UUU','VVV','XXX','YYY','ZZZ']
#statnames = ['A1' ,'A2' ,'A3' ,'A4' ,'A5' ,'A6' ,'B1' ,'B2' ,'B3' ,'B4' ,'B5' ,'B6' ,'C1' ,'C2' ,'C3' ,'C4' ,'C5' ,'C6' ,'D1' ,'D2' ,'D3' ,'D4' ,'D5' ,'D6' ,'E1' ,'E2' ,'E3' ,'E4' ,'E5' ,'E6' ,'F1' ,'F2' ,'F3' ,'F4' ,'F5' ,'F6']
#statnames = ['A1' ,'A2' ,'A3' ,'A4' ,'B1' ,'B2' ,'B3' ,'B4' ,'C1' ,'C2' ,'C3' ,'C4' ,'D1' ,'D2' ,'D3' ,'D4']
statnames = ['A1' ,'A2' ,'A3' ,'A4' ,'A5' ,'A6' , 'A7', 'B1' ,'B2' ,'B3' ,'B4' ,'B5' ,'B6' , 'B7','C1' ,'C2' ,'C3' ,'C4' ,'C5' ,'C6' ,'C7','D1' ,'D2' ,'D3' ,'D4' ,'D5' ,'D6' ,'D7','E1' ,'E2' ,'E3' ,'E4' ,'E5' ,'E6' ,'E7','F1' ,'F2' ,'F3' ,'F4' ,'F5' ,'F6','F7','G1' ,'G2' ,'G3' ,'G4' ,'G5' ,'G6','G7']


GV=['.090','.000','.ver']

distance=np.zeros((nRec,nShot))
for ishot in range(1,nShot+1):
    
    for i,statname in enumerate(statnames):     

        print('ireceiver='+str(i))
        distance[i,ishot-1]=((R[i,1]-S[ishot-1,1])**2+(R[i,2]-S[ishot-1,2])**2+(R[i,0]-S[ishot-1,0])**2)**(0.5)

f_err = open('Distance.dat','w')
distance.astype('float').tofile(f_err)     
#print(Err)    
    
    
    
