ZOOM=3
DATA=data/unders
GET_US () { echo -n $(head -n1 data/undersampling_factor_${1}.txt) | tr '.' '-' ; }  
WMAX=0.5
CFLCOMMON="-z$ZOOM -u$WMAX -FZ -x2 -y1"

NEWTON=11
REDU=2

NLINV_OPTS="-a240 -b40 -i${NEWTON} -R${REDU} -S"
SAKE_ITER=50
DNLINV_OPTS="--noise_sigma 1.0 --model_type unet --num_samples 4 --infer_num_samples 500 --lr 1e-3 --n_iter 12000 --img_shape 192 192 --network_params 32 4 --dropout_prob 0.0 --calib_region_shape 1 1 --seed 34433 --enable_cosine_scheduler --h5_type bart --noise_covariance_estimate --n_mps 1 --weight_decay 1e-6 --coil_init_max_iter 150 --image_init_max_iter 150 --coil_num_upsample 2 --l2_reg 0"
DIP_OPTS="--noise_sigma 1.0 --model_type unet --infer_num_samples 500 --lr 1e-3 --n_iter 1500 --img_shape 192 192 --network_params 32 4 --dropout_prob 0.0 --calib_region_shape 1 1 --seed 34433 --enable_cosine_scheduler --h5_type bart --noise_covariance_estimate --n_mps 1 --weight_decay 1e-6 --coil_init_max_iter 150 --image_init_max_iter 150 --coil_num_upsample 2 --l2_reg 0 --maximum_likelihood"

# Undersampling values
VALS=(  #value  corresponding US factor
        1.205   # 4.0
#        1.573   # 5.0
#        1.86    # 6.0
        2.111   # 7.0
#        2.25    # 7.5
#        2.34    # 8.0
        2.45    # 8.5
 #       2.56    # 9.0
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
	bart poisson -Y192 -Z192 -y$US -z$US -v -V10 -s$SEED data/tmp_pat_${US} > /dev/null
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

