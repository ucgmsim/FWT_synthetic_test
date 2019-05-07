#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
import numpy as np
import os
import time
from qcore import timeseries as ts
#from numpy.linalg import solve
#import math
#import matplotlib.pyplot as plt
#import pdb; 

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

def write_par_source_i(Si):
    sname='FWD/e3d_mysource_i.par'
    #file_default='e3d_mysource.par'
    os.system('cp FWD/e3d_mysource_xyz_default.par FWD/e3d_mysource_i.par')
    fid=open(sname,'a')
    fid.write("%s\n" %('xsrc='+str(int(Si[0]))))
    fid.write("%s\n" %('ysrc='+str(int(Si[1]))))
    fid.write("%s\n" %('zsrc='+str(int(Si[2]))))

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

def Vsp_read(nx,ny,nz,snap_file):
    fid=open(snap_file,'r')
    sek=np.fromfile(fid, dtype='<f4')
    Vp=np.reshape(sek,[ny,nz,nx])
    return Vp

def write_par_opt_source_i(Si):
    fname='FWD/e3d_mysource_i_opt.par'
    #file_default='e3d_mysource.par'
    os.system('cp FWD/e3d_opt_default.par FWD/e3d_mysource_i_opt.par')
    fid=open(fname,'a')
    fid.write("%s\n" %('xsrc='+str(int(Si[0]))))
    fid.write("%s\n" %('ysrc='+str(int(Si[1]))))
    fid.write("%s\n" %('zsrc='+str(int(Si[2]))))

# Read_Dev_strain_ex2.m
nx=200
ny=200
nz=100
nts=5000
#nt=500

#dt=0.02

source_file='../../../StatInfo/SOURCE.txt'
nShot, S = read_source(source_file)

os.system('rm -r ../../Vel_es/*')
#os.system('rm -r All_shots/*')
#os.system('rm -r ../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/*')

mkdir_p('../../Vel_ob/Vel_ob_i')
mkdir_p('../../Vel_es/Vel_es_i')
mkdir_p('Dump')
############################################################
#os.system('rm ../../../Model/Models/vs3dfile_in.s')
#os.system('rm ../../../Model/Models/vp3dfile_in.p')
#os.system('rm ../../../Model/Models/rho3dfile_in.d')

#os.system('cp ../../../../Model/Models/vs3dfile_in.s ../../../Model/Models/vs3dfile_in.s')
#os.system('cp ../../../../Model/Models/vp3dfile_in.p ../../../Model/Models/vp3dfile_in.p')
#os.system('cp ../../../../Model/Models/rho3dfile_in.d ../../../Model/Models/rho3dfile_in.d')
#time.sleep(10)
fi1=open('ipart.dat','r')
ipart=np.int64(np.fromfile(fi1,dtype='int64'))
fi1.close()    
ipart=int(ipart)
os.system('rm -r ../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/*')
#for ishot in range(1,nShot+1):
for ishot in range((ipart-1)*int(nShot/4)+1,ipart*int(nShot/4)+1):
    #dir_es='../../Vel_es/Vel_es_'+str(ishot)
    #mkdir_p(dir_es)
    #dir_opt='../../Vel_opt/Vel_ob_'+str(ishot)
    #mkdir_p(dir_opt)

    fi=open('iShot.dat','w')
    (np.int64(ishot)).tofile(fi)
    fi.close() 
    print('isource='+str(ishot))
    #os.system('rm -r ../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/*')
    os.system('rm ../../Dev_Strain/*')

    os.system('cp ../../../../Kernels/Vel_ob/Vel_ob_'+str(ishot)+'/*.* ../../Vel_ob/Vel_ob_i/')       
    write_par_source_i(S[ishot-1,:])
    
    job_file11 = 'FWT_emod3d_shot_i_part1.sl'
    os.system("sbatch %s" %job_file11)
    time.sleep(160)
#python -c "from qcore import timeseries as ts; lf = ts.LFSeis('$FWD/OutBin'); lf.all2txt(prefix='$Vel_es_i/')"    
#    ts.LFSeis('../../../FwdSims/V3.0.7-xyz/OutBin').all2txt(prefix='../../Vel_es/Vel_es_i/')

    job_file = 'adj_test.sl'
    os.system("sbatch %s" %job_file)
    time.sleep(15)
    #input('Press 1 then Enter to continue')
    job_file12 = 'FWT_emod3d_shot_i_part2.sl'
    os.system("sbatch %s" %job_file12)
    time.sleep(160)

    os.system('rm -r ../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/*')
    job_file2 = 'kernel_shot_i_iter1.sl'
    os.system("sbatch %s" %job_file2)
    time.sleep(220)
    os.system('mv ../../Vel_es/Vel_es_i/*.* ../../../../Kernels/Vel_es/Vel_es_'+str(ishot))          
    os.system('mv KS.txt ../../../../Kernels/Iters/iter1/All_shots/GS_shot'+str(ishot)+'.txt')
    os.system('mv KP.txt ../../../../Kernels/Iters/iter1/All_shots/GP_shot'+str(ishot)+'.txt')



    
