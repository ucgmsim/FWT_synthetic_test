#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
import numpy as np
import numpy.linalg as la
#import os
#import time

def read_srf_source(source_file):

    with open(source_file, 'r') as f:
        lines = f.readlines()
    line0=lines[0].split()
    nShot=int(line0[0])
    S=np.zeros((nShot,3))
    sNames = []
    for i in range(1,nShot+1):
        line_i=lines[i].split()
        S[i-1,0]=line_i[0]
        S[i-1,1]=line_i[1]
        S[i-1,2]=line_i[2]
        sNames.append(line_i[3])

    return nShot, S, sNames

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

def geo_correlation(nShot,nRec,R,S):
    
    Cij=np.zeros((nShot,nRec))   
 
    dh=2.0
    h0=40.0/dh
    v0=20.0/dh          
        
    
    for i in range(0,nShot):
        for j in range(0,nRec):
            
            for k in range(0,nShot):
                for l in range(0,nRec):              
                    
#                    hs = np.min([1, np.sqrt((S[i,1]-S[k,1])**2+(S[i,0]-S[k,0])**2)/h0])
#                    hr = np.min([1, np.sqrt((R[j,1]-R[l,1])**2+(R[j,0]-R[l,0])**2)/h0])
#                    
#                    vs = np.min([1, np.abs(S[i,2]-S[k,2])/v0])
#                    vr = np.min([1, np.abs(R[j,2]-R[l,2])/v0])
#                    #print([i, j,k ,l])
#                    hsr = np.min([1, np.sqrt((S[i,1]-R[l,1])**2+(S[i,0]-R[l,0])**2)/h0])
#                    hrs = np.min([1, np.sqrt((R[j,1]-S[k,1])**2+(R[j,0]-S[k,0])**2)/h0])
#                    
#                    vsr = np.min([1, np.abs(S[i,2]-R[l,2])/v0])
#                    vrs = np.min([1, np.abs(R[j,2]-S[k,2])/v0])
                    
                    hs = np.min([1, la.norm([S[i,0:2],S[k,0:2]])/h0])
                    hr = np.min([1, la.norm([R[j,0:2],R[l,0:2]])/h0])

                    vs = np.min([1, la.norm([S[i,2],S[k,2]])/v0])
                    vr = np.min([1, la.norm([R[j,2],R[l,2]])/v0])
                    #print([i, j,k ,l])
                    hsr = np.min([1, la.norm([S[i,0:2],R[l,0:2]])/h0])
                    hrs = np.min([1, la.norm([R[j,0:2],S[k,0:2]])/h0])

                    vsr = np.min([1, la.norm([S[i,2],R[l,2]])/v0])
                    vrs = np.min([1, la.norm([R[j,2],S[k,2]])/v0])
                    
                    
                    #Cijkl[i*(nRec-1)+j, k*(nRec-1)+l] = hs*hr*vs*vr + hsr*hrs*vsr*vrs
                    Cij[i,j] = Cij[i,j]+np.cos(np.pi/2*hs)*np.cos(np.pi/2*hr)*np.cos(np.pi/2*vs)*np.cos(np.pi/2*vr) + np.cos(np.pi/2*hsr)*np.cos(np.pi/2*hrs)*np.cos(np.pi/2*vsr)*np.cos(np.pi/2*vrs)                                          
                    
            print('i,j='+str(i)+','+str(j)+', cij='+str(Cij[i,j]))                        
            
    return Cij
#Main script#############
source_file='../../../StatInfo/SOURCE.txt'
#source_file='SOURCE.txt'
nShot, S, sNames = read_srf_source(source_file)

station_file = '../../../StatInfo/STATION.txt'    
#station_file = 'STATION.txt'  
nRec, R, statnames = read_stat_name(station_file)

M = nShot*nRec
Cij = geo_correlation(nShot,nRec,R,S)
#np.savetxt('Dump/geo_correlation.txt', Cij.reshape(-1))
print('finish creating Geometrical correlation')

    
