#!/bin/csh

#Submit this script with: sbatch thefilename

#SBATCH --time=1:00:00   # walltime
##SBATCH --nodes=4    # number of nodes (40? cores per node)
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=80G   # memory per CPU core
#SBATCH --error=master_it.e   # stderr file
#SBATCH --output=master_it.o   # stdout file
#SBATCH --exclusive
##SBATCH --hint=nomultithread

###***#SBATCH --job-name=nr02   # job name
###***#SBATCH --partition=   # queue to run in

echo "Starting" $SLURM_JOB_ID `date`
echo "Initiated on `hostname`"
echo ""
cd "$SLURM_SUBMIT_DIR"           # connect to working directory of sbatch

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

#srun python ../../Read_GP_opt_calc_err_NZ_stations.py
#srun python ../../Read_GP_opt_calc_err_NZ_stations_sorted.py
#srun python ../../Read_GP_opt_calc_err_NZ_stations_sorted_rm_sources.py

#srun python ../../Read_GP_opt_calc_err_NZ_stations_flexwin.py
#srun python ../../Read_GP_opt_calc_err_NZ_stations_flexwin_corrected_T_end.py
srun python ../../Read_GP_opt_calc_err_NZ_stations_pyflex_all.py

echo "Done" `date`
exit
