#!/bin/csh

#Submit this script with: sbatch thefilename

#SBATCH --time=1:00:00   # walltime
##SBATCH --nodes=4    # number of nodes (40? cores per node)
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=80G   # memory per CPU core
#SBATCH --error=test_adj.e   # stderr file
#SBATCH --output=test_adj.o   # stdout file
#SBATCH --exclusive
##SBATCH --hint=nomultithread

###***#SBATCH --job-name=nr02   # job name
###***#SBATCH --partition=   # queue to run in

echo "Starting" $SLURM_JOB_ID `date`
echo "Initiated on `hostname`"
echo ""
cd "$SLURM_SUBMIT_DIR"           # connect to working directory of sbatch

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

#srun python ../../Read_GP_adj_NZ_stations.py
#srun python ../../Read_GP_adj_NZ_stations_sorted.py
#srun python ../../Read_GP_adj_NZ_stations_displacement_Tape_flexwin.py

#srun python ../../Read_GP_adj_NZ_stations_sorted_wpk.py
#srun python ../../Read_GP_adj_NZ_stations_sorted_wpk_015Hz.py
#srun python ../../Read_GP_adj_NZ_stations_sorted_wpk_01Hz.py
#srun python ../../Read_GP_adj_NZ_stations_sorted_wpk_02Hz_r2.py

#srun python ../../Read_GP_adj_NZ_stations_sorted_Alan_flexwin.py
srun python ../../Read_GP_adj_NZ_stations_sorted_pyflex.py

#srun python ../../Read_GP_adj_NZ_stations_sorted_Tape_flexwin.py
#srun python ../../Read_GP_adj_NZ_stations_sorted_wpk_01Hz_flexwin.py
#srun python ../../Read_GP_adj_NZ_stations_sorted_Tape_flexwin_corrected_T_end.py

echo "Done" `date`
exit