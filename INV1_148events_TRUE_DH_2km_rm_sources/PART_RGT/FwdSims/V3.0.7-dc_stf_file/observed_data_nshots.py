#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
import numpy as np
import os
import time
#import math
#import matplotlib.pyplot as plt
#import pdb; 
def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

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

def write_par_obs_source_i(Si):
    fname='e3d_mysource_i_obs.par'
    #file_default='e3d_mysource.par'
    os.system('cp e3d_obs.par e3d_mysource_i_obs.par')
    fid=open(fname,'a')
    fid.write("%s\n" %('xsrc='+str(Si[0])))
    fid.write("%s\n" %('ysrc='+str(Si[1])))
    fid.write("%s\n" %('zsrc='+str(Si[2])))

# Read_Dev_strain_ex2.m
nx=65
ny=65
#nz=75
nz=40
nts=10000
nt=500

dx=2.0
dy=2.0
dz=2.0
#dt=0.02

source_file='../../StatInfo/SOURCE.txt'
nShot, S, sNames = read_srf_source(source_file)

os.system('rm OutBin/*.*')
os.system('rm -r ../../Kernels/Vel_ob_ref/*')

mkdir_p('../../Kernels/Vel_ob_ref')
mkdir_p('../../Kernels/Vel_ob_ref/Vel_ob_i')
############################################################
    
for ishot in range(1,nShot+1):
    dir_ob='../../Kernels/Vel_ob_ref/Vel_ob_'+str(ishot)
    mkdir_p(dir_ob)
        
    write_par_obs_source_i(S[ishot-1,:])
    #os.system('cp ../../../Srf/'+str(sNames[ishot-1])+'.srf Srf/srf_file.srf')
    #input('-->')
    job_file1 = 'fdrun-mpi_mysource_i_obs.sl'
    os.system("sbatch %s" %job_file1)
    time.sleep(90)
    os.system('mv ../../Kernels/Vel_ob_ref/Vel_ob_i/*.* ../../Kernels/Vel_ob/Vel_ob_'+str(ishot))
    #input('-->') 
print('Finish Generate Data')
#####################################
#time.sleep(20)


    
