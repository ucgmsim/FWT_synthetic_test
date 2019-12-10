#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:28:15 2019

@author: user
"""

import numpy as np
import os
import time
from numpy.linalg import solve
from shared_workflow.shared import exe
from datetime import datetime
import shlex
from subprocess import Popen, PIPE
from numpy import linalg as LA

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

################################################################
def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

def read_source_new(source_file):
    
    with open(source_file, 'r') as f:
        lines = f.readlines()    
    line0=lines[0].split()
    nShot=int(line0[0])
    S=np.zeros((nShot,3))
    sNames=[]
    for i in range(1,nShot+1):
        line_i=lines[i].split()
        S[i-1,0]=line_i[0]
        S[i-1,1]=line_i[1]
        S[i-1,2]=line_i[2]
        sNames.append(line_i[3])
    
    return nShot, S, sNames 


def Step_length_CMT():
    
    source_file = '../../../../StatInfo/SOURCE.txt'
    nShot, S, sNames = read_source_new(source_file)
    
    fi=open('iShot.dat','r')
    iShot=np.int64(np.fromfile(fi,dtype='int64'))
    fi.close()
    iShot=int(iShot)
    Sloc =S[iShot,:]
    print(Sloc)
        
    fi1=open('iter_s.dat','r')
    iter_s=np.int64(np.fromfile(fi1,dtype='int64'))
    fi1.close()
    iter_s=int(iter_s)
    print('iter_s='+str(iter_s))    
    
    grad_M_arr1 = np.loadtxt('Dump_source/Grad_M.iter'+str(iter_s)+'.txt')
    M0_arr = np.loadtxt('Dump_source/M0.iter'+str(iter_s))
    
    gra_M=grad_M_arr1.reshape([3,3])
    M0=M0_arr.reshape([3,3])

    f_err=open('cmt_err.dat.it'+str(iter_s),'r'); Err0=np.fromfile(f_err,dtype=np.float64); print('Err='+str(Err0));    
    
    L20=Err0
    L21=1.1*Err0
    L22=0	
    flag=0
    Mu20=0
    Mu21=0
    Mu22=0	
    alt_step=0

    st=0.05
    count=0
    iter_max=2
          
    while (L21>L20 and count<iter_max and flag==0):
   
        if (count>0):
            st=st/2	

        M0_new = M0 - st*LA.norm(M0)/LA.norm(gra_M)*gra_M
        print('M0_new ='+str(M0_new))
        print('Update the srf and run forward simulation with perturbed CMT model')        
        input('-->')
        job_file11 = 'FWT_emod3d_shot_i_part1_stf.sl'
        job_submittted(job_file11)
        
        job_file12 = 'no_adj_cmt.sl'
        job_submittted(job_file12)        
        
        f_err=open('cmt_err.dat','r'); Err1=np.fromfile(f_err,dtype=np.float64); os.system("rm err_opt.dat")
        L21=Err1
        Mu21=st
        print('L21='+str(L21))
        count=count+1

        if (Mu21<0.02):
            flag=4
            print('flag='+str(flag))
            opt_step_L2=Mu21
            print('st='+str(st))             
            
#####            
    if (L21<L20 or Mu21<0.02):
        alt_step=Mu21
    ###
    count=0
    while (L21>L22 and flag==0):
        st=st*1.5
        M0_new = M0 - st*LA.norm(M0)/LA.norm(gra_M)*gra_M
        print('M0_new ='+str(M0_new))
        print('Update the srf and run forward simulation with perturbed CMT model')        
        input('-->')
        job_file11 = 'FWT_emod3d_shot_i_part1_stf.sl'
        job_submittted(job_file11)
        
        job_file12 = 'no_adj_cmt.sl'
        job_submittted(job_file12)
        
        f_err=open('cmt_err.dat','r'); Err1=np.fromfile(f_err,dtype=np.float64); os.system("rm err_opt.dat")

        L22=Err1
        print('L22='+str(L22))
        Mu22=st
        
        if(count>0):
            alt_step=Mu22
        
        count=count+1
        if (count==iter_max) or (Mu22>0.1):
            flag=2
            opt_step_L2=st
        
    if (flag==0):
        A = [[Mu20**2, Mu20, 1], [Mu21**2, Mu21, 1], [Mu22**2, Mu22, 1]]    
        B = [L20,L21,L22]
        print(A)
        print(B)
        X=solve(A,B)
        opt_step_L2 = -X[1]/(2*X[0])
        if (opt_step_L2<0):
            opt_step_L2=Mu20
        if (opt_step_L2>Mu22):
            opt_step_L2=Mu22
        
#Confirm optimal step length
    print('Confirm optimal step length')
    st=opt_step_L2
    
    M0_new = M0 - st*LA.norm(M0)/LA.norm(gra_M)*gra_M
    print('M0_new ='+str(M0_new))
    print('Update the srf and run forward simulation with perturbed CMT model')        
    input('-->')
    job_file11 = 'FWT_emod3d_shot_i_part1_stf.sl'
    job_submittted(job_file11)
    
    job_file12 = 'no_adj_cmt.sl'
    job_submittted(job_file12)
    
    f_err=open('cmt_err.dat','r'); Err2=np.fromfile(f_err,dtype=np.float64); 
    print('Err2='+str(Err2))
        
    if (Err2>Err0):
        st=alt_step
        print('alt_step='+str(alt_step))
        M0_new = M0 - st*LA.norm(M0)/LA.norm(gra_M)*gra_M
        flag=3
##############################            
    print('flag=')
    print(flag)
    
    print('write updated cmt model & optimal step length')
    np.savetxt('Dump_source/M0.iter'+str(iter_s+1),M0_new.reshape(-1))
    fi=open('Dump_source/step_cmt_iter_'+str(iter_s)+'.dat','w'); (np.float64(st)).tofile(fi); fi.close()
    
    return opt_step_L2,flag     
