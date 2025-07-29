#!/bin/bash -l

#$ -P ace-ig          # SCC project name
#$ -l h_rt=03:00:00   # Specify the hard time limit for the job
#$ -N I               # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -pe omp 8
#$ mem_per_core=4G

module load python3/3.8.10
module load pytorch/1.13.1

source /projectnb/ace-genetics/jueqiw/software/venvs/monai/bin/activate

cd /project/ace-genetics/jueqiw/code/Autism_Brain_Development/code/project/

python 2D_MRI_VAE_regression.py \
    --recon_weight 10.0 \
    --kl_weight 0.005 \
    --age_weight 1.0 \
    --latent_dim 64 \
    --learning_rate 0.0005 \
    --batch_size 32 \
    --n_epochs 200 \
    --stage_transition_epoch 50 \
    --experiment_name "vae_better_reconstruction_v1_with_conditional"

# python3 /project/ace-genetics/jueqiw/code/Autism_Brain_Development/code/project/weighted_l1_all_monai_autoencoder.py \
#     --loss_type simple_l1 \
#     --weight_clamp_max 20.0 \
#     --experiment_name "simple_l1_weight_clamp_max_20.0" \
#     --batch_size 64

# python3 /project/ace-ig/jueqiw/code/Autism_Brain_Development/code/project/create_2D_image.py --batch_size 64