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

fi1=open('ipart.dat','r')
ipart=np.int64(np.fromfile(fi1,dtype='int64'))
fi1.close()    
ipart=int(ipart)
os.system('rm -r ../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/*')
for ishot in range(16,16+1):

    fi=open('iShot.dat','w')
    (np.int64(ishot)).tofile(fi)
    fi.close() 
    print('isource='+str(ishot))
    os.system('rm -r ../../../../Kernels/Vel_es/Vel_es_'+str(ishot)+'/*')
    os.system('rm -r ../../../../Kernels/Iters/iter1/All_shots/*'+str(ishot)+'.txt')
    os.system('rm ../../Dev_Strain/*')

    os.system('cp ../../../../Kernels/Vel_ob/Vel_ob_'+str(ishot)+'/*.* ../../Vel_ob/Vel_ob_i/')       
    write_par_source_i(S[ishot-1,:])
    
    job_file11 = 'FWT_emod3d_shot_i_part1.sl'
    #os.system("sbatch %s" %job_file11)
#    time.sleep(300)
    #track time of submit

    submit_time = datetime.now().strftime('%Y-%m-%d')
    jobid = submit_script(job_file11)
    wait_job_to_finish(jobid,submit_time)
    print("fw emod3d finished")
#    sys.exit()
#python -c "from qcore import timeseries as ts; lf = ts.LFSeis('$FWD/OutBin'); lf.all2txt(prefix='$Vel_es_i/')"    
#    ts.LFSeis('../../../FwdSims/V3.0.7-xyz/OutBin').all2txt(prefix='../../Vel_es/Vel_es_i/')

    job_file = 'adj_test.sl'
#    os.system("sbatch %s" %job_file)
#    time.sleep(15)
    submit_time = datetime.now().strftime('%Y-%m-%d')
    jobid = submit_script(job_file)
    wait_job_to_finish(jobid,submit_time)
    print("adjoint source calculation finished")


#input('Press 1 then Enter to continue')
    job_file12 = 'FWT_emod3d_shot_i_part2.sl'
#    os.system("sbatch %s" %job_file12)
#    time.sleep(300)
    submit_time = datetime.now().strftime('%Y-%m-%d')
    jobid = submit_script(job_file12)
    wait_job_to_finish(jobid,submit_time)
    print("bw emod3d finished")


    os.system('rm -r ../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/*')

    job_file2 = 'kernel_shot_i_iter1.sl'
#    os.system("sbatch %s" %job_file2)
#    time.sleep(450)
    submit_time = datetime.now().strftime('%Y-%m-%d')
    jobid = submit_script(job_file2)
    wait_job_to_finish(jobid,submit_time)
    print("kernel calculation finished")

    os.system('mv ../../Vel_es/Vel_es_i/*.* ../../../../Kernels/Vel_es/Vel_es_'+str(ishot))          
    os.system('mv KS.txt ../../../../Kernels/Iters/iter1/All_shots/GS_shot'+str(ishot)+'.txt')
    os.system('mv KP.txt ../../../../Kernels/Iters/iter1/All_shots/GP_shot'+str(ishot)+'.txt')



    
