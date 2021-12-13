# 2D ablation experiments with DNLINV

1. Do regular DNLINV reconstruction (0_dnlinv_recon.sh)
2. Do DNLINV reconstruction w/o noise estimation (1_dnlinv_fixed_noise.py) - run with same arguments
3. Do DNLINV reconstruction with linear activations (2_dnlinv_linear_act.py) - run with same arguments
4. Do DIP (3_dip.sh) - run with maximum likelihood
5. Do DIP with variance and MC-inference (4_dip_mc.py)  - run with maximum likelihood and also need to set --dip_stdev to non-zero
6. Collect results (5_plot_results.py)


Shell scripts are files that call the original `reconstruct.py` with the modes already supported by input arguments to 
the code. Other modes require internal modifications and the code is based on the original `reconstruct.py` with 
specific modifications marked with the comment: `2D ABLATION MODIFICATION`
You may validate this by taking the `diff` of the files.

A SLURM high-performance computing cluster-compatible script is provided in `slurm_doit.sh`. Call this script
with the command `srun <slurm_arguments> slurm_doit.sh` or `sbatch <slurm_arguments> slurm_doit.sh`.
