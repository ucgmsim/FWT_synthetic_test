#!/bin/csh

#Submit this script with: sbatch thefilename

#SBATCH --time=1:00:00   # walltime
##SBATCH --nodes=4    # number of nodes (40? cores per node)
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=80G   # memory per CPU core
#SBATCH --error=test_sum.e   # stderr file
#SBATCH --output=test_sum.o   # stdout file
#SBATCH --exclusive
##SBATCH --hint=nomultithread

###***#SBATCH --job-name=nr02   # job name
###***#SBATCH --partition=   # queue to run in

echo "Starting" $SLURM_JOB_ID `date`
echo "Initiated on `hostname`"
echo ""
cd "$SLURM_SUBMIT_DIR"           # connect to working directory of sbatch

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

#srun python Sum_Gradients_nShot_vsp_tape_xyz_corrected.py 
#srun python Sum_Gradients_nShot_vsp_geometry_corrected.py

#srun python Sum_Gradients_nShot_vsp_tape_xyz_flexwin.py
srun python Sum_Gradients_nShot_vsp_tape_xyz_pyflex.py

echo "Done" `date`
exit
