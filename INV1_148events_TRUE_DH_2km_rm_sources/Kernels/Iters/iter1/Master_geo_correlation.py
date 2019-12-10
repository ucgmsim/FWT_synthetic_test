#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
import numpy as np
import os
import time
#from numpy.linalg import solve
import Step_length_srf_checked
from shared_workflow.shared import exe
from datetime import datetime
import shlex
from subprocess import Popen, PIPE
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

def job_submittted(job_file):
        submit_time = datetime.now().strftime('%Y-%m-%d')
        jobid = submit_script(job_file)
        wait_job_to_finish(jobid,submit_time)        

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

def write_par_source_i(Si):
    sname='FWD/e3d_mysource_i.par'
    #file_default='e3d_mysource.par'
    os.system('cp FWD/e3d_mysource_xyz_default.par FWD/e3d_mysource_i.par')
    fid=open(sname,'a')
    fid.write("%s\n" %('xsrc='+str(Si[0])))
    fid.write("%s\n" %('ysrc='+str(Si[1])))
    fid.write("%s\n" %('zsrc='+str(Si[2])))

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

def write_par_opt_source_i(Si):
    fname='FWD/e3d_mysource_i_opt.par'
    #file_default='e3d_mysource.par'
    os.system('cp FWD/e3d_opt_default.par FWD/e3d_mysource_i_opt.par')
    fid=open(fname,'a')
    fid.write("%s\n" %('xsrc='+str(Si[0])))
    fid.write("%s\n" %('ysrc='+str(Si[1])))
    fid.write("%s\n" %('zsrc='+str(Si[2])))
    
def write_slurm_part_i(ipart):
    fname='fwi_part_i.sl'
    #file_default='e3d_mysource.par'
    os.system('cp fwi_part_i_default.sl fwi_part_i.sl')
    fid=open(fname,'a')
#    fid.write("%s\n" %('srun python ../../../PART'+str(ipart)+'/Kernels/Iters/iter1/Master_iterations_part_i.py'))
    fid.write("%s\n" %('srun python Master_iterations_part_i.py'))
    fid.write("%s\n" %('echo "Done" `date`'))
    fid.write("%s\n" %('exit'))

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

def Bcos(x):
    if x>1:
        Bx=0
    else:
        Bx=np.cos(np.pi/2*x)
    
    return Bx   

def geo_correlation(nShot,nRec,R,S):
    
    M = nShot*nRec
    Cijkl = np.zeros((M,M))
    wr = np.zeros((M,1))
    
    hs = np.zeros((nShot,nShot))
    hr = np.zeros((nRec,nRec))   
    vs = np.zeros((nShot,nShot))
    vr = np.zeros((nRec,nRec))       
    
    hsr = np.zeros((nShot,nRec))
    hrs = np.zeros((nRec,nShot))   
    vsr = np.zeros((nShot,nRec))
    vrs = np.zeros((nRec,nShot)) 
 
    dh=2.0
    h0=40.0/dh
    v0=20.0/dh          
        
    
    for i in range(0,nShot):
        for j in range(0,nRec):
            
            for k in range(0,nShot):
                for l in range(0,nRec):              
                    
                    hs[i,k] = Bcos(np.sqrt((S[i,1]-S[k,1])**2+(S[i,0]-S[k,0])**2)/h0)
                    hr[j,l] = Bcos(np.sqrt((R[j,1]-R[l,1])**2+(R[j,0]-R[l,0])**2)/h0)
                    
                    vs[i,k] = Bcos(np.abs(S[i,2]-S[k,2])/v0)
                    vr[j,l] = Bcos(np.abs(R[j,2]-R[l,2])/v0)
                    #print([i, j,k ,l])
                    hsr[i,l] = Bcos(np.sqrt((S[i,1]-R[l,1])**2+(S[i,0]-R[l,0])**2)/h0)
                    hrs[j,k] = Bcos(np.sqrt((R[j,1]-S[k,1])**2+(R[j,0]-S[k,0])**2)/h0)
                    
                    vsr[i,l] = Bcos(np.abs(S[i,2]-R[l,2])/v0)
                    vrs[j,k] = Bcos(np.abs(R[j,2]-S[k,2])/v0)
                    
                    Cijkl[i*(nRec-1)+j, k*(nRec-1)+l] = hs[i,k]*hr[j,l]*vs[i,k]*vr[j,l] + hsr[i,l]*hrs[j,k]*vsr[i,l]*vrs[j,k]
            
            wr[i*(nRec-1)+j] = np.sum(Cijkl[i*(nRec-1)+j, :])                                        
                        
    return wr
#Main script#############
source_file='../../../StatInfo/SOURCE.txt'
nShot, S, sNames = read_srf_source(source_file)

station_file = '../../../StatInfo/STATION.txt'    
nRec, R, statnames = read_stat_name(station_file)

M = nShot*nRec
wr = geo_correlation(nShot,nRec,R,S)
np.savetxt('Dump/geo_correlation.txt', wr)
print('finish creating Geometrical correlation')

    
