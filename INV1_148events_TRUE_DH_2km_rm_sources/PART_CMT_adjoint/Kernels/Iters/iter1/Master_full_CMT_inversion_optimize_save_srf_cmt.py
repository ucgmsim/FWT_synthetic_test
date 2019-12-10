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
from Step_length_srf_pyflex_CMT_iterative_nosrf import Step_length_CMT
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

def job_submittted(job_file):
        submit_time = datetime.now().strftime('%Y-%m-%d')
        jobid = submit_script(job_file)
        wait_job_to_finish(jobid,submit_time)             

################################################################

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

def read_vs_1d(vname, depth):
    """
    Read the vs_1d and rho_1d from lp_generic1d-gp01_v1.vmod
    """    
    with open(vname, 'r') as f:
        lines = f.readlines() 
    line0=lines[0].split()
    n_col = int(line0[0])
        
    data = []

    for line in lines[1:]:
        data.append([float(val) for val in line.split()])

    data = np.concatenate(data)         
    v_mod = data.reshape([n_col,6])
    
    depth_ref = 0
    for i in range(0, n_col):
        depth_ref = depth_ref+v_mod[i,0]
        #print(depth_ref)
        if(depth_ref>depth):
            vs_1d = v_mod[i-1,2]
            rho_1d = v_mod[i-1,3]        
            break
    
    return vs_1d, rho_1d

def read_srf_new(fname):
    """
    Convinience function for reading files in the Graves and Pitarka format
    """
    with open(fname, 'r') as f:
        lines = f.readlines()

    line5=lines[5].split()
    line6=lines[6].split()

    depth=float(line5[2])
    strike=float(line5[3])
    dip=float(line5[4])
    rake=float(line6[0])
    
    slip=float(line6[1])
    area=float(line5[5])
#    Vs_1d=float(line5[8])
#    rho_1d=float(line5[9])
    vs_1d, rho_1d = read_vs_1d('/scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/tdn27/rgraves/NZVMs/Christchurch_Events/lp_generic1d-gp01_v1.vmod', depth)
    
    rigid = (vs_1d*10**5)*(vs_1d*10**5)*rho_1d
    Mo = slip*area*rigid
    
    print(['{:.5e}'.format(slip), '{:.5e}'.format(area), vs_1d, rho_1d, '{:.2e}'.format(Mo)])
    
    return depth, strike, dip, rake, Mo

def sdr2cmt(strike, dip, rake, Mo):
    
    #Assume that rot=0 or Mxx=Mnn otherwise Mxx = Mnn*cosA*cosA + Mee*sinA*sinA + Mne*sin2A;

    PI = np.pi*1/180.0 # Convert from degree to radian
    Mzz=np.sin(2*dip*PI)*np.sin(rake*PI)
    Mxx=np.sin(dip*PI)*np.cos(rake*PI)*np.sin(2*strike*PI) + np.sin(2*dip*PI)*np.sin(rake*PI)*np.sin(strike*PI)**2
    Mxx=-Mxx

    Myy=np.sin(dip*PI)*np.cos(rake*PI)*np.sin(2*strike*PI) - np.sin(2*dip*PI)*np.sin(rake*PI)*np.cos(strike*PI)**2
    Mxz=np.cos(dip*PI)*np.cos(rake*PI)*np.cos(strike*PI) + np.cos(2*dip*PI)*np.sin(rake*PI)*np.sin(strike*PI)
    Mxz=-Mxz

    Myz=np.cos(dip*PI)*np.cos(rake*PI)*np.sin(strike*PI) - np.cos(2*dip*PI)*np.sin(rake*PI)*np.cos(strike*PI)
    Mxy=np.sin(dip*PI)*np.cos(rake*PI)*np.cos(2*strike*PI) + 0.5*np.sin(2*dip*PI)*np.sin(rake*PI)*np.sin(2*strike*PI)
    Myz=-Myz

    M9 = Mo*np.array([[Mxx, Mxy, Mxz],[Mxy, Myy, Myz],[Mxz, Myz, Mzz]])

    return M9


def write_par_full_CMT_source_i(Si,M_full):
    
    M = M_full
    fname='FWD/e3d_mysource_i_full_CMT.par'
    #file_default='e3d_mysource.par'
    os.system('cp FWD/e3d_full_CMT_default.par FWD/e3d_mysource_i_full_CMT.par')
    fid=open(fname,'a')
    
    fid.write("%s\n" %('Mnn='+str('{:.8e}'.format(M[0,0]))))
    fid.write("%s\n" %('Mee='+str('{:.8e}'.format(M[1,1]))))
    fid.write("%s\n" %('Mdd='+str('{:.8e}'.format(M[2,2]))))
    fid.write("%s\n" %('Mne='+str('{:.8e}'.format(M[0,1]))))
    fid.write("%s\n" %('Mnd='+str('{:.8e}'.format(M[0,2]))))
    fid.write("%s\n" %('Med='+str('{:.8e}'.format(M[1,2]))))

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
############################################################

R_all_arr=np.loadtxt('../../../../Kernels/index_all_ncc_pyflex_gt05_V2_excluded.txt')
R_all=R_all_arr.reshape([nRec,3,nShot])

#ishot_arr=[1, 5, 28, 37, 55, 98, 120, 69]
#ishot_arr=[1, 5]
ishot_arr=[1, 5, 28, 37, 55, 98, 120, 69]

#for ishot_id in range(1,2):
for ishot_id in range(1,len(ishot_arr)):
    ishot=ishot_arr[ishot_id]

    fi=open('iShot.dat','w')
    (np.int64(ishot)).tofile(fi)
    fi.close() 
    print('isource='+str(ishot))    
    
    os.system('cp ../../../../Kernels/Vel_ob_new_V2/Vel_ob_'+str(ishot)+'/*.* ../../Vel_ob_new_V2/Vel_ob_i/')     
    #os.system('cp ../../../Kernels/Vel_test/*.* ../../Vel_ob_new_V2/Vel_ob_i/')
   
    for iter_s in range(0,5):
        fi=open('iter_s.dat','w')
        (np.int64(iter_s)).tofile(fi)
        fi.close()
        
        print('iter_s='+str(iter_s))
      
        os.system('rm -r ../../Vel_es/Vel_es_i/*')
        os.system('rm -r ../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/*')    
      
        if(iter_s==0):
            os.system('cp /scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/tdn27/rgraves/NZVMs/Christchurch_Events/Srf/'+str(sNames[ishot-1])+'.srf srf_file.srf')
            _, strike, dip, rake, Mo = read_srf_new('srf_file.srf')
            M_full = sdr2cmt(strike, dip, rake, Mo)
            print(M_full)
            np.savetxt('Dump_source/M0.it'+str(iter_s)+'.txt',M_full.reshape(-1))
            #input('-->')
        else:
            #input('Load CMT to continue:')
            M0_arr=np.loadtxt('Dump_source/M0.it'+str(iter_s)+'.txt')
            M_full=M0_arr.reshape([3,3])
            print(M_full)


        write_par_full_CMT_source_i(S[ishot-1,:],M_full)
        
#        input('-->')   
        job_file11 = 'FWT_emod3d_shot_i_full_CMT_part1_stf.sl'
        job_submittted(job_file11)
    #    print("fw emod3d finished")

#        input('adjoint simulation-->')
    
        job_file = 'adj_cmt.sl'
        job_submittted(job_file)
#        os.system('python ../../Read_GP_adj_NZ_stations_sorted_pyflex_cmt_new.py')
        time.sleep(5)
        print("adjoint source calculation finished") 
        #input('adjoint simulation-->')
        #os.system('cat test_adj.o')   
        os.system('mv cmt_err.dat Dump_source/cmt_err.dat.it'+str(iter_s))
         
        job_file12 = 'FWT_emod3d_shot_i_part2.sl'
        job_submittted(job_file12)
    #    print("bw emod3d finished")

        os.system('rm -r ../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/*')
    #input('Press 1 then Enter to continue')
        job_file2 = 'kernel_shot_i_CMT_nsource_nosrf.sl'
        job_submittted(job_file2)
        print("cmt kernel calculation finished")
    #    time.sleep(5)
   
        opt_step_L2,flag = Step_length_CMT()
        print("cmt optimization finished")

        f_err=open('cmt_err.dat','r'); Err2=np.fromfile(f_err,dtype=np.float64); print('Err2='+str(Err2))    
        os.system('cp cmt_err.dat cmt_err.dat.it'+str(iter_s))

        if((iter_s>1) and (flag>2)):
            break

    os.system('cp Dump_source/M0.it'+str(iter_s+1)+'.txt Dump_source/'+str(sNames[ishot-1])+'.srf.cmt.it'+str(iter_s+1)+'.txt')
    print('save Dump_source/'+str(sNames[ishot-1])+'.srf.cmt.update')
    #os.system('cat test_cmt.o')

           


    
