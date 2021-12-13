COMMON_OPTS="--noise_sigma 0.01 --model_type unet --infer_num_samples 1000 --lr 1e-4 --img_shape 256 256 --network_params 32 4 --dropout_prob 0.0 --batch_norm_momentum 0.1 --undersampling_type parallel_imaging_y --undersampling_factor 8 --calib_region_shape 1 1 --seed 34433 --n_mps 1 --weight_decay 1e-12 --coil_init_max_iter 0 --image_init_max_iter 0 --coil_fov_oversampling 1.0 --coil_num_upsample 1 --l2_reg 0 --existing_mps phantom --disable_data_scaling"
DNLINV_OPTS="$COMMON_OPTS"
DIP_OPTS="$COMMON_OPTS --maximum_likelihood"
SENSE_OPTS="--seed 34433 --noise_sigma 0.01 --img_shape 256 256 --undersampling_type parallel_imaging_y --undersampling_factor 8 --calib_region_shape 1 1 --seed 34433 --weight_decay 0.5e-2 --existing_mps phantom --disable_data_scaling"

ITERS=(
  10
  50
  100
  250
  500
  1000
  2500
  5000
  7500
  10000
  12500
  15000
  20000
)
