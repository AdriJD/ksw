#!/bin/bash                                                                                                                                                    
#SBATCH --job-name=fisher                                                                                                                                       
#SBATCH --nodes=1
#SBATCH --ntasks=1                                                                                                                                                                                                                                                                             
#SBATCH --cpus-per-task=16                                                                                                                                                                                                                                                                               
#SBATCH --mem=512G
#SBATCH --time=02:00:00   


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /home1/p319540/miniconda3/etc/profile.d/conda.sh
conda activate base

ANADIR=${HOME}/ksw/tests/python
ODIR=${HOME}/ksw/tests/python


cd ${ANADIR}
srun -u --cpu-bind=cores -c $SLURM_CPUS_PER_TASK python test_fr_radial_func_dL_tensor.py --odir ${ODIR} 

# cp ./slurm-${SLURM_JOB_ID}.out ${ODIR}/