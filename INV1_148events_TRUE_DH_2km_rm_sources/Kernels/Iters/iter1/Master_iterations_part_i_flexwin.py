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

def write_par_opt_source_i(Si):
    fname='FWD/e3d_mysource_i_opt.par'
    #file_default='e3d_mysource.par'
    os.system('cp FWD/e3d_opt_default.par FWD/e3d_mysource_i_opt.par')
    fid=open(fname,'a')
    fid.write("%s\n" %('xsrc='+str(int(Si[0]))))
    fid.write("%s\n" %('ysrc='+str(int(Si[1]))))
    fid.write("%s\n" %('zsrc='+str(int(Si[2]))))

# Read_Dev_strain_ex2.m
dt=0.08

station_file = '../../../../StatInfo/STATION.txt'    
nRec, R, statnames = read_stat_name(station_file)

source_file='../../../../StatInfo/SOURCE.txt'
nShot, S, sNames = read_srf_source(source_file)

os.system('rm -r ../../Vel_es/*')
#os.system('rm -r All_shots/*')
#os.system('rm -r ../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/*')

mkdir_p('../../Vel_ob_new_V2/')
mkdir_p('../../Vel_ob_new_V2/Vel_ob_i')
mkdir_p('../../Vel_es/Vel_es_i')
mkdir_p('Dump')
############################################################

fi1=open('ipart.dat','r')
ipart=np.int64(np.fromfile(fi1,dtype='int64'))
fi1.close()    
ipart=int(ipart)
os.system('rm -r ../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/*')

#R_all_arr=np.loadtxt('../../../../Kernels/index_all_ncc_gt005.txt')
#R_all_arr=np.loadtxt('../../../../Kernels/index_all_ncc_gt005_t_end_corrected.txt')
#R_all_arr=np.loadtxt('../../../../Kernels/index_all_ncc_gt005_tshift_20s_new.txt')
#R_all_arr=np.loadtxt('../../../../Kernels/index_all_ncc_gt005_V2.txt')
R_all_arr=np.loadtxt('../../../../Kernels/index_all_ncc_gt005_V2_10s.txt')
R_all=R_all_arr.reshape([nRec,3,nShot])

#for ishot in range(,16+1):
for ishot in range((ipart-1)*int(nShot/4)+1,ipart*int(nShot/4)+1):
#    R_ishot_arr=np.loadtxt('../../../../Kernels/Iters/iter1/Dump/R_ishot_'+str(ishot)+'.txt')
    if ((S[ishot-1,2]<20) and (np.sum(R_all[:,:,ishot-1])>12)):
        fi=open('iShot.dat','w')
        (np.int64(ishot)).tofile(fi)
        fi.close() 
        print('isource='+str(ishot))
        os.system('rm -r ../../../../Kernels/Vel_es/Vel_es_'+str(ishot)+'/*')
        os.system('rm -r ../../../../Kernels/Iters/iter1/All_shots/GS_shot'+str(ishot)+'.txt')
        os.system('rm -r ../../../../Kernels/Iters/iter1/All_shots/GP_shot'+str(ishot)+'.txt')    
        os.system('rm ../../Dev_Strain/*')
        os.system('rm -r ../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/*')
    
        os.system('cp ../../../../Kernels/Vel_ob_new_V2/Vel_ob_'+str(ishot)+'/*.* ../../Vel_ob_new_V2/Vel_ob_i/')       
        write_par_source_i(S[ishot-1,:])
        os.system('cp /scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/tdn27/rgraves/NZVMs/Christchurch_Events/Srf/'+str(sNames[ishot-1])+'.srf srf_file.srf')
    
        
        job_file11 = 'FWT_emod3d_shot_i_part1.sl'
        #os.system("sbatch %s" %job_file11)
    #    time.sleep(300)
        #track time of submit
    
        submit_time = datetime.now().strftime('%Y-%m-%d')
        jobid = submit_script(job_file11)
        wait_job_to_finish(jobid,submit_time)
    #    print("fw emod3d finished")
    #    sys.exit()
    #python -c "from qcore import timeseries as ts; lf = ts.LFSeis('$FWD/OutBin'); lf.all2txt(prefix='$Vel_es_i/')"    
    #    ts.LFSeis('../../../FwdSims/V3.0.7-xyz/OutBin').all2txt(prefix='../../Vel_es/Vel_es_i/')
    
        job_file = 'adj_test.sl'
    #    os.system("sbatch %s" %job_file)
    #    time.sleep(15)
        submit_time = datetime.now().strftime('%Y-%m-%d')
        jobid = submit_script(job_file)
        wait_job_to_finish(jobid,submit_time)
    #    print("adjoint source calculation finished")
    
    
    #input('Press 1 then Enter to continue')
        job_file12 = 'FWT_emod3d_shot_i_part2.sl'
    #    os.system("sbatch %s" %job_file12)
    #    time.sleep(300)
        submit_time = datetime.now().strftime('%Y-%m-%d')
        jobid = submit_script(job_file12)
        wait_job_to_finish(jobid,submit_time)
    #    print("bw emod3d finished")
    
    
    #    os.system('rm -r ../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/*')
    
        job_file2 = 'kernel_shot_i_iter1.sl'
    #    os.system("sbatch %s" %job_file2)
    #    time.sleep(450)
        submit_time = datetime.now().strftime('%Y-%m-%d')
        jobid = submit_script(job_file2)
        wait_job_to_finish(jobid,submit_time)
    #    print("kernel calculation finished")
    #    time.sleep(5)
        os.system('mv ../../Vel_es/Vel_es_i/*.* ../../../../Kernels/Vel_es/Vel_es_'+str(ishot))          
        os.system('mv KS.txt ../../../../Kernels/Iters/iter1/All_shots/GS_shot'+str(ishot)+'.txt')
        os.system('mv KP.txt ../../../../Kernels/Iters/iter1/All_shots/GP_shot'+str(ishot)+'.txt')
        time.sleep(5)
        
    else:
        print('No kernel calculated for source '+str(ishot))             


    
