# Calibrationless reconstruction experiments using 3T proton-density fat-suppressed knee data from the fastMRI knee challenge

Run `doit.sh` in the command-line or run through the separate shell scripts in each directory sequentially.

A SLURM high-performance computing cluster-compatible script is provided in `slurm_doit.sh` under each directory. Call this script
with the command `srun <slurm_arguments> slurm_doit.sh` or `sbatch <slurm_arguments> slurm_doit.sh`. Alternatively, you may simply
run `run_slurm_batch_job.sh` to call `sbatch` in each of the directories. Make sure to update the `SLURM_OPTS` 
line in `run_slurm_batch_job.sh` to the options available to your SLURM cluster.
