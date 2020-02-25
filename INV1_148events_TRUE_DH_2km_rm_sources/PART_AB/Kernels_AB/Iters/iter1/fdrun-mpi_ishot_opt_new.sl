#!/bin/csh

#Submit this script with: sbatch thefilename

#SBATCH --time=1:00:00   # walltime
###-#SBATCH --nodes=4    # number of nodes (40? cores per node)
#SBATCH --ntasks=320   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH --error=test.e   # stderr file
#SBATCH --output=test.o   # stdout file
#SBATCH --exclusive

###***#SBATCH --job-name=nr02   # job name
###***#SBATCH --partition=   # queue to run in

echo "Starting" $SLURM_JOB_ID `date`
echo "Initiated on `hostname`"
echo ""
cd "$SLURM_SUBMIT_DIR"           # connect to working directory of sbatch

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

set VERSION = 3.0.7-dev
set EMOD_DIR = /home/rgraves/Mpi/Emod3d/V3.0.7-dev
#set EMOD_DIR = ${HOME}/Mpi/Emod3d/V${VERSION}
set OUTBIN = ../../../FwdSims/V3.0.7-dc_stf_file/OutBin
set Vel_ob_i = ../../../Kernels/Vel_opt/Vel_ob_i
set FWD = ../../../FwdSims/V3.0.7-dc_stf_file

set MY_RUN_ID = `echo $user $SLURM_JOB_ID | gawk '{split($2,a,".");printf "%s-%s\n",$1,a[1];}'`
echo "MY_RUN_ID=" $MY_RUN_ID
#set NP = ` ./set_runparams_v2.csh `
###-limit stacksize 10240
#srun -n $NP $EMOD_DIR/emod3d-mpi par=e3d.par < /dev/null

#python update model

srun $EMOD_DIR/emod3d-mpi par=FWD/e3d_mysource_i_opt.par < /dev/null
echo "emod3d 1 finished"
python -c "from qcore import timeseries as ts; ts.LFSeis('$OUTBIN').all2txt(prefix='$Vel_ob_i/')"
echo "winbin aio finished"

#python calculate err

echo "Done" `date`
exit