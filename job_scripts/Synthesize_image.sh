module load python3/3.10.5

python3 -m pip install disptools


module load ants/2.6.2

antsApplyTransforms --float --default-value 0  \
--input /projectnb/ace-genetics/jueqiw/dataset/MRI_template/Atlas/BNA-maxprob-thr0-1mm.nii.gz \
--reference-image /projectnb/ace-genetics/jueqiw/dataset/MRI_template/Space/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz \
-t /projectnb/ace-genetics/jueqiw/dataset/MRI_template/Space/tpl-MNI152NLin2009cAsym_from-MNI152NLin6Asym_mode-image_xfm.h5 \
--interpolation NearestNeighbor  -d 3 -e 3 \
--output /projectnb/ace-genetics/jueqiw/dataset/MRI_template/Atlas/BN_Atlas_246_space-MNI152NLin2009cAsym.nii.gz


antsApplyTransforms --float --default-value 0  \
--input /projectnb/ace-genetics/jueqiw/dataset/MRI_template/Atlas/BNA-maxprob-thr0-1mm.nii.gz \
--reference-image /projectnb/ace-genetics/jueqiw/dataset/MRI_template/Space/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz \
-t /Users/a16446/Documents/GitHub/Autism_Brain_Development/data/MRI_template/Space/tpl-MNI152NLin2009cAsym_from-MNI152NLin6Asym_mode-image_xfm.h5 \
--interpolation NearestNeighbor  -d 3 -e 3 \
--output /projectnb/ace-genetics/jueqiw/dataset/MRI_template/Atlas/BN_Atlas_246_space-MNI152NLin2009cAsym.nii.gz


antsApplyTransforms --float --default-value 0  \
--input /Users/a16446/Documents/GitHub/Autism_Brain_Development/data/MRI_template/Atlas/BNA-maxprob-thr0-1mm.nii.gz \
-r /Users/a16446/.cache/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz \
-t /Users/a16446/.cache/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_from-MNI152NLin6Asym_mode-image_xfm.h5 \
--interpolation NearestNeighbor  -d 3 -e 3 \
--output /Users/a16446/Documents/GitHub/Autism_Brain_Development/data/MRI_template/Atlas/BN_Atlas_246_space-MNI152NLin2009cAsym.nii.gz