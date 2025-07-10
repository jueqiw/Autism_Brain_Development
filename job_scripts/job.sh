#!/bin/bash -l

#$ -P ace-ig          # SCC project name
#$ -l h_rt=03:00:00   # Specify the hard time limit for the job
#$ -N AD_try          # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -l gpus=1
#$ -l gpu_c=6.0

module load python3/3.8.10
export PYTHONPATH=/projectnb/ace-ig/jw_python/lib/python3.8.10/site-packages:$PYTHONPATH
module load pytorch/1.13.1

python3 /project/ace-ig/jueqiw/code/Autism_Brain_Development/code/project/create_2D_image.py

python3 /project/ace-ig/jueqiw/code/CrossModalityLearning/code/project/smriprep_rest_images.py




python3 /project/ace-ig/jueqiw/code/CrossModalityLearning/code/project/smriprep_ABIDE_II.py

python3 /project/ace-ig/jueqiw/code/CrossModalityLearning/code/project/create_bids_folder_ABIDE_II.py

python3 /project/ace-ig/jueqiw/code/CrossModalityLearning/code/project/create_bids_folder.py




python3 encoder_genetics.py \
	--dataset=SSC \
	--model=Genetics_Encoder \
	--experiment_name="ACE_test"


python3 /project/ace-ig/jueqiw/code/CrossModalityLearning/code/project/main.py \
	--tensor_board_logger="/projectnb/ace-ig/jueqiw/experiment/CrossModalityLearning/tensorboard/" \
	--dataset="ACE" \
	--experiment_name="ACE_lr_5e-5_ld_8_log_z_qk_32_q_32_k_4_v_8_KL_sparsity_1e-2_1e-6_Softsign_0.5_with_pathway_paired_loss_weight_0.0001_final" \
	--classifier_latent_dim=32 \
	--batch_size=128 \
	--learning_rate=5e-5 \
	--test_fold=0 \
	--run_time=0 \
	--n_epochs=1501 \
	--hidden_dim_qk=24 \
	--hidden_dim_q=32 \
	--hidden_dim_k=4 \
	--hidden_dim_v=8 \
	--relu_at_coattention \
	--normalization="batch" \
	--bernoulli_probability=1e-1 \
	--sparsity_loss_weight=1e-6 \
	--soft_sign_constant=0.5 \
	--contrastive_metric="L1" \
	--diff_pair_loss \
	--pair_loss_weight=0.00001 \
	--not_write_tensorboard






import os

for i in range(10):
     os.system(f"qsub with_pair_loss_{i}.sh")