#!/bin/bash -l

#$ -P ace-ig          # SCC project name
#$ -l h_rt=03:00:00   # Specify the hard time limit for the job
#$ -N ACE_only_gene   # Give job a name
#$ -j y               # Merge the error and output streams into a single file

module load python3/3.8.10
export PYTHONPATH=/projectnb/ace-ig/jw_python/lib/python3.8.10/site-packages:$PYTHONPATH
module load pytorch/1.13.1

source /projectnb/ace-genetics/grace/tf_env/bin/activate

python3 /project/ace-ig/jueqiw/code/Autism_Brain_Development/code/project/train_img.py
