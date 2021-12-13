import os
import os.path
import subprocess
import glob

fast_mri_data_path = "D:\\Box\\dev\\data\\fastMRI_multichannel\\multicoil_val\\pdfs"

# Hard-coded data files for 3T PD contrast data
with open("3t_pdfs_data_files.csv", 'r') as f:
    filenames = f.read()
data_files = filenames.split("\n")

# Remove any empty strings
data_files = [f for f in data_files if len(f) > 0]

dnlinv_path = os.environ['DNLINV_PATH']
fastmri_conversion_script_path = os.path.join(dnlinv_path, "fastmri_convert_h5_to_cfl.py")

processing_scripts = "0_reco_dnlinv.sh 1_reco_enlive.sh 2_reco_fastMRI_unet.sh 3_get_measurements.sh 4_create_us_fig.sh doit.sh slurm_doit.sh opts.sh"

for idx, f in enumerate(data_files):
    print(f"Processing {f}")

    h5_path = os.path.join(fast_mri_data_path, f)

    p_dir = f"P{idx + 1}"
    out_dir = os.path.join(p_dir, "data")
    if os.path.exists(out_dir) is not True:
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, "slice-full")
    run_command = f"python {fastmri_conversion_script_path} {h5_path} {out_path}" 

    print(run_command)
    subprocess.run(run_command)

    subprocess.run(f"cp {processing_scripts} {os.path.join(p_dir, '.')}")  # Copy reconstruction scripts
    subprocess.run(f"cp gen_pattern.py {os.path.join(p_dir, '.')}")

print('Done setting up data and scripts!')