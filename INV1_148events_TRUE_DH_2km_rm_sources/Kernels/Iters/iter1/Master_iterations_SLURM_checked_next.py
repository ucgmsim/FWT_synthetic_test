#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
import numpy as np
import os
import time
#from numpy.linalg import solve
#import Step_length_srf_checked
import Step_length_srf_checked_rm_sources
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

    h0=10.0
    v0=20.0          
        
    
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
#nx=200
#ny=200
#nz=100
#nts=5000
#nt=500
#
#dx=1.0
#dy=dx
#dz=dx
#dt=0.02

iters=10
#nShot=16
source_file='../../../StatInfo/SOURCE.txt'
nShot, S, sNames = read_srf_source(source_file)

station_file = '../../../StatInfo/STATION.txt'    
nRec, R, statnames = read_stat_name(station_file)

#M = nShot*nRec
#wr = geo_correlation(nShot,nRec,R,S)
#np.savetxt('Dump/geo_correlation.txt', wr)
print('finish creating Geometrical correlation')
#mkdir_p('../../Vel_es/Vel_es_i')
#mkdir_p('../../Vel_ob/Ves_ob_i')
mkdir_p('../../Vel_opt/Vel_ob_i')
mkdir_p('Dump')
############################################################
for it in range(2,iters+1):
    fi=open('iNumber.dat','w')
    (np.int64(it)).tofile(fi)
    fi.close() 
    print('iNumber='+str(it))
#    if (it==1):
#        os.system('cp ../../../Model/Models/vs3dfile_h.s ../../../Model/Models/vs3dfile_in.s')
#        os.system('cp ../../../Model/Models/vp3dfile_h.p ../../../Model/Models/vp3dfile_in.p')
#        os.system('cp ../../../Model/Models/rho3dfile_h.d ../../../Model/Models/rho3dfile_in.d')
#        
#    if (it>1):
#        os.system('cp %s' %'Dump/vs3dfile_iter_'+str(it-1)+'.s ../../../Model/Models/vs3dfile_in.s')
#        os.system('cp %s' %'Dump/vp3dfile_iter_'+str(it-1)+'.p ../../../Model/Models/vp3dfile_in.p')
#        os.system('cp %s' %'Dump/rho3dfile_iter_'+str(it-1)+'.d ../../../Model/Models/rho3dfile_in.d')
#        
#    time.sleep(5) 
#
#    os.system('rm -r All_shots/*')
#    os.system('rm -r ../../Vel_es/*')
#
#    for ishot in range(1,nShot+1):
#        dir_es='../../Vel_es/Vel_es_'+str(ishot)
#        mkdir_p(dir_es)
#
#    for ipart in range(1,8):
#        
#        fi=open('ipart.dat','w')
#        (np.int64(ipart)).tofile(fi)
#        fi.close() 
#        print('ipart='+str(ipart))
#        
#        #os.system('cp -r %s' %'../../Vel_ob/* ../../../PART'+str(ipart)+'/Kernels/Vel_ob/')
#
#        os.system('mv %s' %'ipart.dat ../../../PART'+str(ipart)+'/Kernels/Iters/iter1/')
#        #os.system('cp %s' %'Master_iterations_part_i.py ../../../PART'+str(ipart)+'/Kernels/Iters/iter1/')
#        os.system('cp %s' %'Master_iterations_part_i_checked_corrected.py ../../../PART'+str(ipart)+'/Kernels/Iters/iter1/')
#        os.system('cp %s' %'FWT_emod3d_shot_i_part1.sl.part ../../../PART'+str(ipart)+'/Kernels/Iters/iter1/FWT_emod3d_shot_i_part1.sl')
#        os.system('cp %s' %'FWD/e3d_mysource_xyz_default.par.part ../../../PART'+str(ipart)+'/Kernels/Iters/iter1/FWD/e3d_mysource_xyz_default.par')
#        os.system('cp %s' %'ADJ/set_run+merge_params_new_h.csh.part ../../../PART'+str(ipart)+'/Kernels/Iters/iter1/ADJ/set_run+merge_params_new_h.csh')
#
#        input('part=')
#
#    input('Finish copy data and kernels, then press 1 to continue')
        

###############################################################
    fi1=open('iTape.dat','r')
    R_nf=np.int64(np.fromfile(fi1,dtype='int64'))
    fi1.close()
    print('R_nf='+str(R_nf))    

    job_file3 = 'sum_kernel.sl'
#    job_file3 = 'sum_kernel_4parts.sl'
    job_submittted(job_file3)
    print('finish summing kernels')
    #input('-->')
#    #################################
    job_file4 = 'precondition_gradient.sl'
    job_submittted(job_file4)     
#    os.system("sbatch %s" %job_file4)
#    time.sleep(90)
    print('finish precoditioning kernels')
    beta=np.loadtxt('Dump/Beta_iter_'+str(it)+'.txt')
    print('Beta='+str(beta))
    #input('-->')    
    ##############################    
    #job_file = '../../Read_GP_obs_calc_nShot.py'        
    job_file5 = 'observed_misfit.sl'
    job_submittted(job_file5)   
#    os.system("sbatch %s" %job_file5)
#    time.sleep(85) 
    f_err0=open('err_obs.dat','r'); Err=np.fromfile(f_err0,dtype=np.float64); print('Err='+str(Err))         
    os.system('cp err_obs.dat Dump/err_iter_'+str(it)+'.dat')    
#    input('-->')    
    print('Search for optimal step length')
#    opt_step_L2,flag = Step_length_srf_checked.Step_length(Err,S,nShot,sNames,it)
    opt_step_L2,flag = Step_length_srf_checked_rm_sources.Step_length(Err,S,nShot,sNames,it)
    input('-->')
    print('Finish Update iteration')
    #exit()
    #####################################
    #time.sleep(20)


    
