
# 2D Shepp-Logan phantom reconstruction experiments

Phantom reconstruction relies on the DNLINV reconstruction code with a few modifications to enable stable and reproducible phantom
reconstruction. Sections of modification are marked with `2D PHANTOM MODIFICATION`.

Run `doit.sh` in the command-line or run through the separate shell scripts sequentially.

A SLURM high-performance computing cluster-compatible script is provided in `slurm_reco_phantom.sh`. Call this script
with the command `srun <slurm_arguments> slurm_reco_phantom.sh` or `sbatch <slurm_arguments> slurm_reco_phantom.sh`.
