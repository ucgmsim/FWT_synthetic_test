#!/bin/csh

#Submit this script with: sbatch thefilename

#SBATCH --time=1:00:00   # walltime
###-#SBATCH --nodes=4    # number of nodes (40? cores per node)
#SBATCH --ntasks=320   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH --error=test.e   # stderr file
#SBATCH --output=test.o   # stdout file
#SBATCH --exclusive
##SBATCH --hint=nomultithread

echo "Starting" $SLURM_JOB_ID `date`
echo "Initiated on `hostname`"
echo ""
cd "$SLURM_SUBMIT_DIR"           # connect to working directory of sbatch

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

set VERSION = 3.0.8
set BINDIR = /scale_wlg_persistent/filesets/home/rgraves/Mpi/Emod3d/V${VERSION}
#set BINDIR = /nesi/project/nesi00213/opt/maui/hybrid_sim_tools/emod3d-mpi_v3.0.8

#set EQID = 2012p001403-m4pt8
set EQID = 2012p001403_2km
set ELON =         172.83
set ELAT =         -43.46
set EDEP =          4.00
set ESTK =          222.0
set EDIP =          53.0
set ERAK =          101.0
set EMOM =    1.71e+23

set NP = ` ./set_runparams.csh $EQID $ELON $ELAT $EDEP $ESTK $EDIP $ERAK $EMOM `

srun -n $NP ${BINDIR}/emod3d-mpi par=e3d.par < /dev/null
echo "FW emod3d"

python -c "from qcore import timeseries as ts; ts.LFSeis('OutBin').all2txt(prefix='Vel/')"
echo "postprocessing  finished"
echo "Done" `date`
exit
