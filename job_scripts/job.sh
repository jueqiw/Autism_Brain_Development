#!/bin/bash -l

#$ -P ace-ig          # SCC project name
#$ -l h_rt=03:00:00   # Specify the hard time limit for the job
#$ -N AD_try          # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -l gpus=1
#$ -l gpu_c=6.0

# DEPRECATION: celery 5.0.5 has a non-standard dependency specifier pytz>dev. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of celery or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
# DEPRECATION: nb-black 1.0.7 has a non-standard dependency specifier black>='19.3'; python_version >= "3.6". pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of nb-black or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
# Installing collected packages: typing-extensions
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# datascience 0.17.0 requires coveralls, which is not installed.
# datascience 0.17.0 requires nbsphinx, which is not installed.
# nni 2.10.1 requires filelock<3.12, but you have filelock 3.13.4 which is incompatible.
# pytype 2024.4.11 requires jinja2>=3.1.2, but you have jinja2 2.11.3 which is incompatible.
# jupyter-server 1.6.4 requires anyio<3,>=2.0.2, but you have anyio 4.4.0 which is incompatible.
# Successfully installed typing-extensions-4.10.0


module load python3/3.8.10
module load pytorch/1.13.1

source /projectnb/ace-genetics/jueqiw/software/venvs/monai/bin/activate

cd /project/ace-genetics/jueqiw/code/Autism_Brain_Development/code/project

python3 /project/ace-genetics/jueqiw/code/Autism_Brain_Development/code/project/weighted_l1_all_monai_autoencoder.py \
    --experiment_name "focal_loss_scale_50_new" \
    --focal_scale_factor 50.0 \
    --loss_type focal_l1 \
    --focal_alpha 1.0

python3 weighted_l1_all_monai_autoencoder.py --experiment_name "weighted_l1" --loss_type balanced_weighted_l1