ZOOM=3
DATA=data/unders
WMAX=0.5
CFLCOMMON="-z$ZOOM -u$WMAX -FZ -x2 -y1"
NEWTON=5
REDU=3

NLINV_OPTS="-a240 -b30 -i${NEWTON} -R${REDU} -S"
DNLINV_OPTS="--noise_sigma 0.01 --model_type unet --infer_num_samples 500 --num_samples 2 --lr 1e-3 --n_iter 12000 --img_shape 320 320 --network_params 32 4 --dropout_prob 0.0 --calib_region_shape 1 1 --seed 34433 --h5_type bart --n_mps 1 --weight_decay 1e-6 --coil_init_max_iter 00 --image_init_max_iter 0 --coil_fov_oversampling 1.25 --coil_num_upsample 1 --l2_reg 0 --enable_cosine_scheduler"

# Undersampling values
USs=(  #value  corresponding US factor
    2
    3
    4		#4
		#8		#8
)
