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

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

def write_par_opt_source_i(Si):
    fname='FWD/e3d_mysource_i_opt.par'
    #file_default='e3d_mysource.par'
    os.system('cp FWD/e3d_opt_default.par FWD/e3d_mysource_i_opt.par')
    fid=open(fname,'a')
    fid.write("%s\n" %('xsrc='+str(Si[0])))
    fid.write("%s\n" %('ysrc='+str(Si[1])))
    fid.write("%s\n" %('zsrc='+str(Si[2])))


def Step_length(Err,S,nShot,sNames,it):
    
    #L2=np.zeros((3,1))
    #Mu2=np.zeros((3,1))
    L20=Err
    L21=1.1*Err
    L22=0	
    flag=0
    Mu20=0
    Mu21=0
    Mu22=0	
    alt_step=0
    
    st=0.1
    count=0
    iter_max=8
       
    while (L21>L20 and count<iter_max ):
   
        if (count>0):
            st=st/2	
            
        fi=open('st.dat','w'); (np.float64(st)).tofile(fi); fi.close() 
        print('st='+str(st))
        job_file = 'model_optimizing.sl'
        os.system("sbatch %s" %job_file)        
        time.sleep(45)
        print('forward simulation with perturbed model')        
        for ishot in range(1,nShot+1):
        #for ishot in range(6,7+1):
            write_par_opt_source_i(S[ishot-1,:])
            os.system('cp /scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/tdn27/rgraves/NZVMs/Christchurch_Events/Srf/'+str(sNames[ishot-1])+'.srf srf_file.srf')
            job_file = 'fdrun-mpi_ishot_opt_new.sl'
            os.system("sbatch %s" %job_file)
            time.sleep(90)
            os.system('mv ../../Vel_opt/Vel_ob_i/*.* ../../Vel_opt/Vel_ob_'+str(ishot))
      
        job_file5 = 'optimized_misfit.sl'
        os.system("sbatch %s" %job_file5)
        print('optimize misfit')
        time.sleep(85)
        f_err=open('err_opt.dat','r'); Err1=np.fromfile(f_err,dtype=np.float64); os.system("rm err_opt.dat")
        L21=Err1
        Mu21=st
        print('L21='+str(L21))
        count=count+1
        if (count==iter_max):
            flag=1
            print('flag='+str(flag))
            opt_step_L2=0
            st=opt_step_L2
            #fi=open('st.dat','w'); (np.float64(st)).tofile(fi); fi.close()
            print('st='+str(st))
            fi=open('Dump/step_iter_'+str(it)+'.dat','w'); (np.float64(st)).tofile(fi); fi.close()
            exit()
            
#####            
    if (L21<L20):
        alt_step=Mu21
    ###
    count=0
    while (L21>L22 and flag==0):
        st=st*1.5
        fi=open('st.dat','w'); (np.float64(st)).tofile(fi); fi.close() 
        print('st='+str(st))
        job_file = 'model_optimizing.sl'
        os.system("sbatch %s" %job_file)  
        time.sleep(45)    
        print('forward simulation with perturbed model')
        for ishot in range(1,nShot+1):
        #for ishot in range(6,7+1):
            #parfile for optimal fw emod3d run
            write_par_opt_source_i(S[ishot-1,:])
            os.system('cp /scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/tdn27/rgraves/NZVMs/Christchurch_Events/Srf/'+str(sNames[ishot-1])+'.srf srf_file.srf')
            job_file = 'fdrun-mpi_ishot_opt_new.sl'
            os.system("sbatch %s" %job_file)
            time.sleep(90)
            os.system('mv ../../Vel_opt/Vel_ob_i/*.* ../../Vel_opt/Vel_ob_'+str(ishot))
            
        job_file5 = 'optimized_misfit.sl'
        os.system("sbatch %s" %job_file5)
        print('optimize misfit')
        time.sleep(85)
        f_err=open('err_opt.dat','r'); Err1=np.fromfile(f_err,dtype=np.float64); os.system("rm err_opt.dat")
        L22=Err1
        print('L22='+str(L22))
        Mu22=st
        
        if(count>0):
            alt_step=Mu22
        
        count=count+1
       # if (count==iter_max) or (Mu22>0.2):
        if (count==iter_max):
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
    fi=open('st.dat','w'); (np.float64(st)).tofile(fi); fi.close() 
    print('st='+str(st))
    job_file = 'model_optimizing.sl'
    os.system("sbatch %s" %job_file)    
    time.sleep(45)
    print('forward simulation with perturbed model')    
    for ishot in range(1,nShot+1):
    #for ishot in range(6,7+1):
        #parfile for optimal fw emod3d run
        write_par_opt_source_i(S[ishot-1,:])
        os.system('cp /scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/tdn27/rgraves/NZVMs/Christchurch_Events/Srf/'+str(sNames[ishot-1])+'.srf srf_file.srf')
        job_file = 'fdrun-mpi_ishot_opt_new.sl'
        os.system("sbatch %s" %job_file)
        time.sleep(90)
        os.system('mv ../../Vel_opt/Vel_ob_i/*.* ../../Vel_opt/Vel_ob_'+str(ishot))
        
    job_file5 = 'optimized_misfit.sl'
    os.system("sbatch %s" %job_file5)
    print('optimize misfit')
    time.sleep(85)
    f_err=open('err_opt.dat','r'); Err2=np.fromfile(f_err,dtype=np.float64); os.system("rm err_opt.dat")
    print('Err2='+str(Err2))
    
    
    if (Err2>Err):
        opt_step_L2=alt_step
        print('alt_step='+str(alt_step))
        flag=3
##############################            
    print('flag=')
    print(flag)
    
    print('write updated model')
    st=opt_step_L2
    fi=open('st.dat','w'); (np.float64(st)).tofile(fi); fi.close() 
    print('st='+str(st))
    job_file = 'model_optimizing.sl'
    os.system("sbatch %s" %job_file)
    time.sleep(45)
    os.system('mv %s' %'Model/vs3dfile_opt.s Dump/vs3dfile_iter_'+str(it)+'.s')
    os.system('mv %s' %'Model/vp3dfile_opt.p Dump/vp3dfile_iter_'+str(it)+'.p')
    os.system('mv %s' %'Model/rho3dfile_opt.d Dump/rho3dfile_iter_'+str(it)+'.d')    
    fi=open('Dump/step_iter_'+str(it)+'.dat','w'); (np.float64(st)).tofile(fi); fi.close()
    
    return opt_step_L2,flag     
