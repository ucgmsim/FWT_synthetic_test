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

def Vsp_read(nx,ny,nz,snap_file):
    fid=open(snap_file,'r')
    sek=np.fromfile(fid, dtype='<f4')
    Vp=np.reshape(sek,[ny,nz,nx])
    return Vp

def write_model_Vsp(rho,Vs,Vp):
    [ny,nz,nx]=Vs.shape
    
    model_file='rho3dfile1.d'
    fid=open(model_file,'wb')
    sek1 = np.array(rho, dtype=np.float32)
    sek1.astype('float32').tofile(fid)
    
    model_file0='vs3dfile1.s'
    model_file1='vp3dfile1.p'

    fid00=open(model_file0,'wb')
    sek2 = np.array(Vs, dtype=np.float32)
    sek2.astype('float32').tofile(fid00)

    fid01=open(model_file1,'wb')
    sek3 = np.array(Vp, dtype=np.float32)
    sek3.astype('float32').tofile(fid01)

def write_par_body_source_i(Si):
    fname='FWD/e3d_mysource_i_opt_1station.par'
    #file_default='e3d_mysource.par'
    os.system('cp FWD/e3d_body_default_1station.par FWD/e3d_mysource_i_opt_1station.par')
    fid=open(fname,'a')
    fid.write("%s\n" %('xsrc='+str(Si[0])))
    fid.write("%s\n" %('ysrc='+str(Si[1])))
    fid.write("%s\n" %('zsrc='+str(Si[2])))

# Read_Dev_strain_ex2.m
nx=65
ny=65
nz=40
nts=2000
nt=500

dx=1.0
dy=dx
dz=dx
#dt=0.02

iters=1
#nShot=16
source_file='../../../StatInfo/SOURCE.txt'
nShot, S, sNames = read_srf_source(source_file)

#os.system('rm -r ../../Vel_es/*')
#os.system('rm -r ../../Vel_adj/*')
#os.system('rm -r ../../Vel_opt/*')
#os.system('rm -r /Dev_Strain/FW_xyz/*')

mkdir_p('../../Vel_es/Vel_es_i')
mkdir_p('../../Vel_adj/Vel_adj_i')
mkdir_p('../../Vel_opt/Vel_ob_i')
mkdir_p('../../Dev_Strain/FW_xyz')
mkdir_p('Dump')

#os.system('rm -r ../../Vel_es/*')
#os.system('rm -r ../../Vel_adj/*')
#os.system('rm -r ../../Vel_opt/*')

os.system('rm -r ../../../../FwdSims/V3.0.7-dc_stf_file/OutBin/*')

############################################################
E=np.zeros((iters,1))
#for it in range(1,iters+1):
for it in range(1,1+1):

    print('iNumber='+str(it))
    if (it==1):
        os.system('cp ../../../../Model/Models/vs3dfile_in.s Model/vs3dfile_opt.s')
        os.system('cp ../../../../Model/Models/vp3dfile_in.p Model/vp3dfile_opt.p')
        os.system('cp ../../../../Model/Models/rho3dfile_in.d Model/rho3dfile_opt.d')
        
    if (it>1):
        os.system('cp %s' %'Dump/vs3dfile_iter_'+str(it-1)+'.s Model/vs3dfile_opt.s')
        os.system('cp %s' %'Dump/vp3dfile_iter_'+str(it-1)+'.p Model/vp3dfile_opt.p')
        os.system('cp %s' %'Dump/rho3dfile_iter_'+str(it-1)+'.d Model/rho3dfile_opt.d')
        
    time.sleep(5) 

    for ishot in range(1,1+1):
#    for ishot in range(1,nShot+1):
#    ishot_arr=[1, 5, 28, 37, 55, 98, 120, 69]
#    for ishot_id in range(0,len(ishot_arr)):
#        ishot=ishot_arr[ishot_id]
        if (it>0):
            dir_es='../../Vel_opt/Vel_ob_'+str(ishot)
            mkdir_p(dir_es)        
        print('isource='+str(ishot))
#        write_par_opt_source_i(S[ishot-1,:])
        #S[ishot-1,:]=[33,33,1]
        S[ishot-1,:]=[33,44,11]
        write_par_body_source_i(S[ishot-1,:])
#        os.system('cp /scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/tdn27/rgraves/NZVMs/Christchurch_Events/Srf/'+str(sNames[ishot-1])+'.srf srf_file.srf')
        job_file = 'fdrun-mpi_ishot_opt_1station.sl'
        job_submittted(job_file)
        os.system('mv ../../../../Kernels/Vel_opt/Vel_ob_i/*.* ../../Vel_opt/Vel_ob_'+str(ishot))
    if(it==1):
        input('copy to window pick-->')           
#    input('Copy est. data, calculate error function and Press 1 to continue')
        
#    job_file = 'optimized_misfit.sl'
#    job_submittted(job_file)
#    f_err0=open('err_opt.dat','r')
#    Err=np.fromfile(f_err0,dtype=np.float64)
#    print('Err='+str(Err))
#    E[it-1]=Err
#    input('next iteration-->')
#print('E='+str(E))         
#f_err = open('err_opt.dat','w')
#E.astype('float').tofile(f_err)
#np.savetxt('err_opt_8iters.txt', E) 
     
    

    
