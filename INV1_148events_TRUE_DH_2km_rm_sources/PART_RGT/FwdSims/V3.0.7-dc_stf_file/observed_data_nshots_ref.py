#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
import numpy as np
import os
import time
from shared_workflow.shared import exe
from datetime import datetime
import shlex
from subprocess import Popen, PIPE
#import math
#import matplotlib.pyplot as plt
#import pdb; 
def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)
        
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
       
        job = out_list[1].split()
        if job[1] == "COMPLETED":
            return 1
        else:
#            print("waiting for job : {} to finish".format(jobid))
            time.sleep(5)
            continue        
        
def job_submittted(job_file):
        submit_time = datetime.now().strftime('%Y-%m-%d')
        jobid = submit_script(job_file)
        wait_job_to_finish(jobid,submit_time)   

###############################################################        

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
dh=2.0

source_file='../../StatInfo/SOURCE.txt'
nShot, S, sNames = read_srf_source(source_file)

os.system('rm OutBin/*.*')
os.system('rm -r ../../Kernels/Vel_ob_ref/*')

mkdir_p('../../Kernels/Vel_ob_ref')
mkdir_p('../../Kernels/Vel_ob/Vel_ob_i')
############################################################
    
for ishot in range(1,nShot+1):
    if (S[ishot-1,2]<20):
        dir_ob='../../Kernels/Vel_ob_ref/Vel_ob_'+str(ishot)
        mkdir_p(dir_ob)
            
        write_par_obs_source_i(S[ishot-1,:])
        os.system('cp ../../../Srf/'+str(sNames[ishot-1])+'.srf Srf/srf_file.srf')
        #input('-->')
        job_file1 = 'fdrun-mpi_mysource_i_obs.sl'
        job_submittted(job_file1)
        os.system('mv ../../Kernels/Vel_ob/Vel_ob_i/*.* ../../Kernels/Vel_ob_ref/Vel_ob_'+str(ishot))
    else:
        print('Remove source '+str(ishot))
    #input('-->') 
print('Finish Generate Data')
#####################################
#time.sleep(20)


    
