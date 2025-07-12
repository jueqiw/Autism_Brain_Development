#!/bin/bash -l

#$ -P ace-ig          # SCC project name
#$ -l h_rt=03:00:00   # Specify the hard time limit for the job
#$ -N AD_try          # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -l gpus=1
#$ -l gpu_c=6.0

DEPRECATION: celery 5.0.5 has a non-standard dependency specifier pytz>dev. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of celery or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
DEPRECATION: nb-black 1.0.7 has a non-standard dependency specifier black>='19.3'; python_version >= "3.6". pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of nb-black or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
Installing collected packages: typing-extensions
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
datascience 0.17.0 requires coveralls, which is not installed.
datascience 0.17.0 requires nbsphinx, which is not installed.
nni 2.10.1 requires filelock<3.12, but you have filelock 3.13.4 which is incompatible.
pytype 2024.4.11 requires jinja2>=3.1.2, but you have jinja2 2.11.3 which is incompatible.
jupyter-server 1.6.4 requires anyio<3,>=2.0.2, but you have anyio 4.4.0 which is incompatible.
Successfully installed typing-extensions-4.10.0


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