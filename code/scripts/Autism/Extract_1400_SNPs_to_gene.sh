module load plink/1.90b6.27
module load plink2/alpha21Nov2023


# +-------------------------------------+
# | 1. Find PRS threshold               |
# +-------------------------------------+
plink2 --bfile /projectnb/ace-ig/jueqiw/dataset/SSC_Pseudo_ACE_onekg/ready_to_merge/ACE/ACE \
      --extract /projectnb/ace-ig/jueqiw/dataset/SSC_Pseudo_ACE_onekg/ready_to_merge/one_kg_LD.valid.snp \
      --threads 8 \
      --make-bed \
      --out /projectnb/ace-ig/jueqiw/dataset/SSC_Pseudo_ACE_onekg/ready_to_merge/ACE/ACE_LD

Rscript /projectnb/ace-ig/jueqiw/software/PRSice_linux/PRSice.R \
    --prsice /projectnb/ace-ig/jueqiw/software/PRSice_linux/PRSice_linux \
    --base /projectnb/ace-ig/jueqiw/dataset/Sayan_ACE_SSC_PRS/Rockfish_Autism_Study/PRS_Analysis/GWAS/ACE.gwas \
    --target /projectnb/ace-ig/jueqiw/dataset/SSC_Pseudo_ACE_onekg/ready_to_merge/ACE/ACE_LD \
    --binary-target T \
    --snp CHR_BP \
    --stat Beta \
    --thread 8 \
    --beta \
    --out /projectnb/ace-ig/jueqiw/dataset/SSC_Pseudo_ACE_onekg/ready_to_merge/ACE/ACE_threshold


# Rscript /projectnb/ace-ig/jueqiw/software/PRSice_linux/PRSice.R \
#     --prsice /projectnb/ace-ig/jueqiw/software/PRSice_linux/PRSice_linux \
#     --base /projectnb/ace-ig/jueqiw/dataset/Sayan_ACE_SSC_PRS/Rockfish_Autism_Study/PRS_Analysis/GWAS/ACE.gwas \
#     --target /projectnb/ace-ig/jueqiw/dataset/SSC_Pseudo_ACE_onekg/ready_to_merge/ACE/ACE_LD \
#     --binary-target T \
#     --snp CHR_BP \
#     --stat Beta \
#     --thread 8 \
#     --beta \
#     --bed A.bed:SetA,B.bed \
#     --multi-plot 10 \
#     --out /projectnb/ace-ig/jueqiw/dataset/SSC_Pseudo_ACE_onekg/ready_to_merge/ACE/ACE_threshold


cd /projectnb/ace-ig/jueqiw/dataset/SSC_ACE_gene/
plink --pfile /projectnb/ace-ig/jueqiw/dataset/SSC_Pseudo_ACE_onekg_call_0.2/merge/ACE_final \
       --threads 8 \
       --extract /project/ace-ig/jueqiw/data/SSC/1400_SNPs_id.txt \
       --make-bed \
       --clump-range \
       --out /projectnb/ace-ig/jueqiw/dataset/SSC_ACE_gene/SSC_1400_SNPs

plink --bfile /projectnb/ace-ig/jueqiw/dataset/SSC_ACE_gene/SSC_1400_SNPs \
      --gene-report /projectnb/ace-ig/jueqiw/dataset/ACE/genetics/prs_analysis/data/GWAS_without_SSC/SSC_ACE.gwas \
      --gene-list glist-hg17 \
      --out /projectnb/ace-ig/jueqiw/dataset/SSC_ACE_gene/SSC_1400_SNPs

plink2 --pfile /projectnb/ace-ig/jueqiw/dataset/SSC_ASD_Pseudo_Controls/RAW/cohorts \
       --extract /projectnb/ace-ig/jueqiw/dataset/SSC_Pseudo_ACE_onekg_call_0.2/merge/one_kg_LD.valid.snp \
       --exclude /projectnb/ace-ig/jueqiw/dataset/SSC_Pseudo_ACE_onekg_call_0.2/merge/SSC_ACE.mismatch \
       --a1-allele /projectnb/ace-ig/jueqiw/dataset/SSC_Pseudo_ACE_onekg_call_0.2/merge/SSC_ACE.a1 \
       --keep /projectnb/ace-ig/jueqiw/dataset/SSC_Pseudo_ACE_onekg_call_0.2/merge/SSC_FID_IID.txt \
       --extract /project/ace-ig/jueqiw/data/SSC/1400_SNPs_id.txt \
       --make-pgen \
       --out /projectnb/ace-ig/jueqiw/dataset/SSC_Pseudo_ACE_onekg_call_0.2/merge/SSC_final

plink --gene-report /projectnb/ace-ig/jueqiw/dataset/ADNI/genetics_data/ADNI_Test_Data/ImputedGenotypes/plink_preprocess/plink/ADNI.1166.snps.gwas \
      --gene-list /projectnb/ace-ig/jueqiw/dataset/ADNI/genetics_data/ADNI_Test_Data/ImputedGenotypes/plink_preprocess/glist-hg18 \
      --gene-list-border 50 \
      --pfilter 0.001 \
      --out plink/ADNI_clean_data_1166_gene