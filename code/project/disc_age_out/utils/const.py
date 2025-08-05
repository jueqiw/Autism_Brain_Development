import os
from pathlib import Path


ABIDE_PATH = Path("/projectnb/ace-ig/ABIDE/").resolve()
ABIDE_I_2D_REGRESSION = Path("/projectnb/ace-ig/ABIDE/ABIDE_I_2D").resolve()
ABIDE_II_2D_REGRESSION = Path("/projectnb/ace-ig/ABIDE/ABIDE_II_2D").resolve()
PRE_TRAINED_WEIGHTS = Path(
    "/projectnb/ace-genetics/jueqiw/experiment/Autism_Brain_Development/pretrain_weight"
).resolve()


ABIDE_I_transform = Path("/projectnb/ace-genetics/ABIDE/ABIDE_I").resolve()
ABIDE_II_transform = Path("/projectnb/ace-genetics/ABIDE/ABIDE_II").resolve()
ABIDE_I_MNI = Path("/projectnb/ace-ig/ABIDE/ABIDE_I_BIDS/derivatives/MNI/").resolve()
ABIDE_II_MNI = Path("/projectnb/ace-ig/ABIDE/ABIDE_II_BIDS/derivatives/MNI/").resolve()
ABIDE_DATA_FOLDER_I = Path("/projectnb/ace-ig/ABIDE/ABIDE_I_ANTS/ABIDE/").resolve()
ABIDE_DATA_FOLDER_I_FREESURFER_RECON = Path(
    "/projectnb/ace-ig/ABIDE/ABIDE_I_freesurfer_recon/ABIDE/"
).resolve()
ABIDE_I_BIDS = Path("/projectnb/ace-ig/ABIDE/ABIDE_I_BIDS").resolve()
ABIDE_II_BIDS = Path("/projectnb/ace-ig/ABIDE/ABIDE_II_BIDS").resolve()
ABIDE_DATA_FOLDER_II = Path("/projectnb/ace-ig/ABIDE/ABIDE_II_T1/ABIDE_II/").resolve()
ABIDE_I_PHENOTYPE = Path("/projectnb/ace-ig/ABIDE/Phenotypic_V1_0b.csv").resolve()
ABIDE_II_PHENOTYPE = Path(
    "/projectnb/ace-ig/ABIDE/ABIDEII_Composite_Phenotypic.csv"
).resolve()
ABIDE_II_PHENOTYPE_Long = Path(
    "/projectnb/ace-ig/ABIDE/ABIDEII_Long_Composite_Phenotypic.csv"
).resolve()
ACE_PHENOTYPE = Path("/projectnb/ace-ig/ace_phenotype.csv").resolve()
TENSORBOARD_LOG_DIR = Path(
    "/projectnb/ace-ig/jueqiw/experiment/CrossModalityLearning/tensorboard"
).resolve()
ACE_FILE = Path(
    "/projectnb/ace-ig/jueqiw/dataset/BrainGenePathway/ACE/final_ACE_KEGG_pathway_with_all_genes_img_4_features_p_threshold_0.1_effect_size_LD_50kb_with_related.csv"
)
ADNI_FILE = Path(
    "/projectnb/ace-ig/jueqiw/dataset/BrainGenePathway/ADNI/Gene/final_AD_KEGG_pathway_with_all_genes_img_p_threshold_0.1_effect_size_LD_50kb.csv"
)
CROSS_VAL_INDEX_ACE = Path(
    "/projectnb/ace-ig/jueqiw/dataset/BrainGenePathway/ACE/10_10_cross_fold_val_index.pkl"
)
CROSS_VAL_INDEX_ADNI = Path(
    "/projectnb/ace-ig/jueqiw/dataset/BrainGenePathway/ADNI/10_10_cross_fold_val_index.pkl"
)
RESULT_FOLDER = Path(
    "/projectnb/ace-ig/jueqiw/experiment/BrainGenePathway/results"
).resolve()

DATA_FOLDER = Path("/projectnb/ace-ig/ABIDE/ABIDE_I_ANTS/ABIDE/").resolve()

TADPOLE_FOLDER = DATA_FOLDER / "tadpole"
P_VALUE_FOLDER = DATA_FOLDER / "genetics_data" / "p_value"
SNP_FOLDER = (
    DATA_FOLDER
    / "genetics_data"
    / "ADNI_Test_Data"
    / "ImputedGenotypes"
    / "plink_preprocess"
)

# ACE data path
# ACE_GENO_FILE = "/projectnb/ace-ig/dataset/ACE/genetics/prs_analysis/data/ACE/GMIND/selected_geno_pheno.csv"
ACE_GENO_FILE = "/projectnb/ace-ig/jueqiw/dataset/ACE/genetics/prs_analysis/data/ACE/GMIND/total_geno.csv"
ACE_IMG_FILE_DESTRIEUX = "/projectnb/ace-ig/jueqiw/dataset/ACE/mri/freesurfer/group_stats/sMRI_destrieux_cortical_thickness_average.csv"
ACE_IMG_FILE_BRAINNETOME = "/projectnb/ace-ig/jueqiw/dataset/ACE/mri/freesurfer/group_stats/ACE_img_Brainnetome.csv"
ACE_IMG_GENO_FOLDER = Path("/project/ace-ig/jueqiw/data/ACE/joint")
