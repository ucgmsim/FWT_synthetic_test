#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
import os
import numpy as np
#import os,sys
import time
#from qcore import timeseries as ts
from shared_workflow.shared import exe
from datetime import datetime
import shlex
from subprocess import Popen, PIPE
#from numpy.linalg import solve
#import math
#import matplotlib.pyplot as plt
#import pdb; 


def submit_script(script):
    res = exe("sbatch {}".format(script), debug=False)
    if len(res[1]) == 0:
        # no errors, return the job id
        return_words = res[0].split()
        job_index = return_words.index("job")
        jobid = return_words[job_index + 1]
        try:
            int(jobid)
        except ValueError:
	        print(
	            "{} is not a valid jobid. Submitting the "
	    	    "job most likely failed".format(jobid)
	        )
        return jobid

def wait_job_to_finish(jobid, time_submited):
    while True:
        #check squeue
        out_list = []
        cmd = "sacct -u tdn27 -S {} -j {} -b ".format(time_submited,jobid)
        #print(cmd)
        process = Popen(shlex.split(cmd), stdout=PIPE, encoding="utf-8")
        (output, err) = process.communicate()
        exit_code = process.wait()

        out_list.extend(filter(None, output.split("\n")[1:]))

        #print(output)
        while len(out_list) <= 1:
            time.sleep(5)
            print('re-querry sacct for jobid : {}'.format(jobid))
            out_list = []
            process = Popen(shlex.split(cmd), stdout=PIPE, encoding="utf-8")
            (output, err) = process.communicate()
            exit_code = process.wait()
            out_list.extend(filter(None, output.split("\n")[1:]))
###############Print out job status###################
#            print(len(out_list))
#        print(out_list)
#        print(len(out_list))
       
        job = out_list[1].split()
        if job[1] == "COMPLETED":
            return 1
        else:
#            print("waiting for job : {} to finish".format(jobid))
            time.sleep(5)
            continue



def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

def write_par_unit_bd_force_source_i(Si,xyz):
    sname='FWD/e3d_mysource_i.par'
    #file_default='e3d_mysource.par'
#    os.system('cp FWD/e3d_mysource_xyz_default_RGT.par FWD/e3d_mysource_i.par')
    os.system('cp FWD/e3d_mysource_xyz_default_RGT_test_Gaussian.par FWD/e3d_mysource_i.par')
    
    fid=open(sname,'a')
    fid.write("%s\n" %('xsrc='+str(int(Si[0]))))
    fid.write("%s\n" %('ysrc='+str(int(Si[1]))))
    fid.write("%s\n" %('zsrc='+str(int(Si[2]))))
    if (xyz==0):
        fid.write("%s\n" %('fxsrc=1.0e+16'))
    if (xyz==1):
        fid.write("%s\n" %('fysrc=1.0e+16'))
    if (xyz==2):
        fid.write("%s\n" %('fzsrc=1.0e+16'))       
        

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

def Vsp_read(nx,ny,nz,snap_file):
    fid=open(snap_file,'r')
    sek=np.fromfile(fid, dtype='<f4')
    Vp=np.reshape(sek,[ny,nz,nx])
    return Vp

# Read_Dev_strain_ex2.m
dt=0.08

station_file = '../../../../StatInfo/STATION_1.txt'    
nRec, R, statnames = read_stat_name(station_file)

#source_file='../../../../StatInfo/SOURCE.txt'
source_file='../../../../StatInfo/STATION_1.txt'
nShot, S, sNames = read_srf_source(source_file)

#os.system('rm -r ../../Vel_es_AB/*')
#os.system('rm -r All_shots/*')
#os.system('rm -r ../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/*')

#mkdir_p('../../Vel_ob_AB/')
#mkdir_p('../../Vel_es_AB/Vel_es_i')
#mkdir_p('Dump')
############################################################

#fi1=open('ipart.dat','r')
#ipart=np.int64(np.fromfile(fi1,dtype='int64'))
#fi1.close()    
#ipart=int(ipart)

#os.system('rm -r ../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/*')

R_all=np.ones([nRec,3,nRec])

#statnames_RGT = ['CBGS','CMHS','DFHS','NNBS','PPHS','REHS']
#statnames_RGT = ['CACS', 'CBGS', 'CRLZ', 'REHS', 'NNBS']
statnames_RGT = ['AAAA']

for i,statname in enumerate(statnames):
    if (statname in statnames_RGT):
        ishot=i
        #mkdir_p('../../Vel_es_AB/Vel_es_'+str(ishot))    
        mkdir_p('../../Dev_Strain/'+statname)
        if (2>1):
            fi=open('iShot.dat','w')
            (np.int64(ishot)).tofile(fi)
            fi.close() 
            print('alpha_receiver='+statname)
            #os.system('rm -r ../../Vel_es_AB/Vel_es_'+str(ishot)+'/*')
            #os.system('rm -r ../../Iters/iter1/All_shots/GS_shot'+str(ishot)+'.txt')
            #os.system('rm -r ../../Iters/iter1/All_shots/GP_shot'+str(ishot)+'.txt')    
            os.system('rm ../../Dev_Strain/*.*')
            #os.system('rm -r ../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/*')
        
#    #        os.system('cp ../../Vel_ob_AB/*.* ../../Vel_ob_new_V2/Vel_ob_i/')       
#            write_par_unit_bd_force_source_i(S[ishot-1,:],0)
#    #        os.system('cp /scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/tdn27/rgraves/NZVMs/Christchurch_Events/Srf/'+str(sNames[ishot-1])+'.srf srf_file.srf')  
#            
#            job_file11 = 'FWT_emod3d_shot_i_RGT.sl'       
#            submit_time = datetime.now().strftime('%Y-%m-%d')
#            jobid = submit_script(job_file11)
#            wait_job_to_finish(jobid,submit_time)
#            print("fw-x finished")
#            mkdir_p('../../Dev_Strain/'+statname+'/RGT_X')
#            os.system('mv ../../Dev_Strain/*.* ../../Dev_Strain/'+statname+'/RGT_X')
##            input('-->')
##            eijp_names = {'exx','eyy','ezz','exy','exz','eyz'};
##            #eijp_names = {'exyp','exzp','eyzp'}
###            fw_file='../../Dev_Strain/fwd01_xyzts.';            
###            matrix_file_fw=fw_file+eij_name            
#
#
#            write_par_unit_bd_force_source_i(S[ishot-1,:],1)
#    #        os.system('cp /scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/tdn27/rgraves/NZVMs/Christchurch_Events/Srf/'+str(sNames[ishot-1])+'.srf srf_file.srf')  
#            
#            job_file11 = 'FWT_emod3d_shot_i_RGT.sl'       
#            submit_time = datetime.now().strftime('%Y-%m-%d')
#            jobid = submit_script(job_file11)
#            wait_job_to_finish(jobid,submit_time)
#            print("fw-y finished")
#            mkdir_p('../../Dev_Strain/'+statname+'/RGT_Y')
#            os.system('mv ../../Dev_Strain/*.* ../../Dev_Strain/'+statname+'/RGT_Y')
            
            write_par_unit_bd_force_source_i(S[ishot,:],2)
    #        os.system('cp /scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/tdn27/rgraves/NZVMs/Christchurch_Events/Srf/'+str(sNames[ishot-1])+'.srf srf_file.srf')  
            print(S[ishot,:])
            input('-->')            
            job_file11 = 'FWT_emod3d_shot_i_RGT.sl'       
            submit_time = datetime.now().strftime('%Y-%m-%d')
            jobid = submit_script(job_file11)
            wait_job_to_finish(jobid,submit_time)
            print("fw-z finished")
            mkdir_p('../../Dev_Strain/'+statname+'/RGT_Z')
            os.system('mv ../../Dev_Strain/*.* ../../Dev_Strain/'+statname+'/RGT_Z')
     


    