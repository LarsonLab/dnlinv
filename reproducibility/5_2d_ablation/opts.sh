DATA=data/unders
GET_US () { echo -n $(head -n1 data/undersampling_factor_${1}.txt) | tr '.' '-' ; }

DNLINV_OPTS="--noise_sigma 1.0 --model_type unet --infer_num_samples 1000 --lr 1e-3 --n_iter 1500 --img_shape 320 320 --network_params 32 4 --dropout_prob 0.0 --calib_region_shape 1 1 --seed 34433 --h5_type bart --noise_covariance_estimate --n_mps 1 --weight_decay 1e-6 --coil_init_max_iter 0 --image_init_max_iter 0 --coil_fov_oversampling 1.5 --coil_num_upsample 1 --l2_reg 0"
DIP_OPTS="--noise_sigma 1.0 --model_type unet --infer_num_samples 1000 --lr 1e-3 --n_iter 1500 --img_shape 320 320 --network_params 32 4 --dropout_prob 0.0 --calib_region_shape 1 1 --seed 34433 --h5_type bart --noise_covariance_estimate --n_mps 1 --weight_decay 1e-6 --coil_init_max_iter 0 --image_init_max_iter 0 --coil_fov_oversampling 1.5 --coil_num_upsample 1 --l2_reg 0 --maximum_likelihood"
DIP_MC_OPTS="--noise_sigma 1.0 --model_type unet --infer_num_samples 1000 --lr 1e-3 --n_iter 1500 --img_shape 320 320 --network_params 32 4 --dropout_prob 0.0 --calib_region_shape 1 1 --seed 34433 --h5_type bart --noise_covariance_estimate --n_mps 1 --weight_decay 1e-6 --coil_init_max_iter 0 --image_init_max_iter 0 --coil_fov_oversampling 1.5 --coil_num_upsample 1 --l2_reg 0 --maximum_likelihood --dip_stdev 1.0"
DNLINV_LINEAR_OPTS="--noise_sigma 1.0 --model_type unet --infer_num_samples 1000 --lr 1e-3 --n_iter 1500 --img_shape 320 320 --network_params 32 4 --dropout_prob 0.0 --calib_region_shape 1 1 --seed 34433 --h5_type bart --noise_covariance_estimate --n_mps 1 --weight_decay 1e-6 --coil_init_max_iter 0 --image_init_max_iter 0 --coil_fov_oversampling 1.5 --coil_num_upsample 1 --l2_reg 0"



# Undersampling values
VALS=(  #value  corresponding US factor
#        1.000   # 2.0
#        1.420   # 3.0
#        1.750   # 4.0
        2.000   # 5.0
#        2.240   # 6.0
#        2.450   # 7.0
#        2.650   # 8.0
#        2.750   # 8.5
)



PREP ()
{
    US=$1
    source opts.sh

    if [ -f data/undersampling_factor_${US}.txt ];
    then
	# prep already done
	return 0
    fi

    SEED=20869

    # VD-PD for total undersampling of $US:
    bart poisson -Y320 -Z256 -y$US -z$US -v -V05 -s$SEED data/tmp_pat_${US} > /dev/null
    bart pattern data/full data/s_${US}
    bart fmac data/tmp_pat_${US} data/s_${US} data/pat_${US}
    cp data/tmp_pat_${US}.cfl data/full_pat_${US}.cfl  # Copy to be used by DNLINV
    cp data/tmp_pat_${US}.hdr data/full_pat_${US}.hdr
    rm data/tmp_pat_${US}.*

    #calculate undersampling
    bart fmac -s 65535 data/s_${US} data/s_${US} data/ns_${US}
    bart fmac -s 65535 data/pat_${US} data/pat_${US} data/npat_${US}
    ALL=$(bart show -f "%+0f%+0fi" data/ns_${US} | cut -f1 -d"." | cut -f2 -d"+")
    PAT=$(bart show -f "%+0f%+0fi" data/npat_${US} | cut -f1 -d"." | cut -f2 -d"+")
    rm data/ns_${US}.* data/npat_${US}.* data/s_${US}.*

    UNDERS=$(echo "scale=1;"$ALL"/"$PAT | bc -l)

    echo $UNDERS > data/undersampling_factor_${US}.txt

    US_NAME=$(GET_US $US)

    mv data/pat_${US}.cfl data/pat_${US_NAME}.cfl
    mv data/pat_${US}.hdr data/pat_${US_NAME}.hdr
    mv data/full_pat_${US}.cfl data/full_pat_${US_NAME}.cfl
    mv data/full_pat_${US}.hdr data/full_pat_${US_NAME}.hdr
    bart fmac data/full data/pat_${US_NAME} data/unders_${US_NAME}
}


GET_DATA ()
{

	if [ -f data/full.cfl ];
	then
		return
	fi

	if false;
	then
		cd data
		wget http://old.mridata.org/knees/fully_sampled/p3/e1/s1/P3.zip
		unzip P3.zip
		cd ..

		#extract single slice
		bart fft -u -i 1 data/kspace data/tmp_fft
		bart slice 0 160 data/tmp_fft data/single_slice
	fi

	bart rss 8 data/single_slice data/tmp_rss
	bart threshold -H 21 data/tmp_rss data/tmp_pat
	bart pattern data/tmp_pat data/pat
	bart fmac data/pat data/single_slice data/tmp_full
	#scale maximum to about 1
	bart scale 1e-8 data/tmp_full data/full


	rm data/tmp_*
}
