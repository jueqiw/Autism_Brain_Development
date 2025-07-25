# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains research code for autism brain development analysis using machine learning approaches. The project combines neuroimaging data (MRI) with genetic pathway information to study autism spectrum disorder (ASD) through various deep learning models including Variational Autoencoders (VAE) and cross-modal attention mechanisms.

## Architecture

### Core Components

**Main Training Scripts:**
- `code/project/project/main.py`: Main training script for brain-pathway analysis with cross-modal attention
- `code/project/2D_MRI_VAE_regression_torch.py`: PyTorch VAE implementation for 2D MRI age regression
- `code/project/weighted_l1_all_monai_autoencoder.py`: MONAI-based autoencoder with weighted L1 loss

**Models Directory (`code/project/project/models/`):**
- `model.py`: BrainPathwayAnalysis - cross-attention model combining MRI and genetic pathway data
- `autoencoder.py`: Various autoencoder architectures
- `genetics_encoder.py`: Encoder specifically for genetic data
- `sMRI_encoder.py`: Structural MRI encoder
- `losses.py`: Custom loss functions including contrastive and pattern losses

**Data Processing:**
- `code/project/project/utils/data_loader.py`: BrainPathwayDataset for multi-modal data loading
- `code/project/utils/transforms.py`: MONAI-based image transformations and augmentations
- `code/project/utils/process_data.py`: Data preprocessing utilities
- `code/project/utils/const.py`: Path configurations and constants

### Data Structure

The project works with several data types:
- **ABIDE I/II datasets**: MRI brain scans from autism research
- **Genetic pathway data**: KEGG pathway information
- **Phenotypic data**: Subject demographics and clinical information
- **Synthesized data**: Generated brain images for data augmentation

### Key Features

1. **Cross-Modal Learning**: Combines structural MRI features with genetic pathway data
2. **Attention Mechanisms**: Cross-attention between brain regions and genetic pathways
3. **VAE Architecture**: Age regression using variational autoencoders
4. **Data Synthesis**: Generation of synthetic brain images for augmentation
5. **Multi-View Processing**: Handles axial, coronal, and sagittal brain slices

## Common Development Tasks

### Core Training Commands

**MONAI-based autoencoder (recommended approach):**
```bash
cd code/project
python weighted_l1_all_monai_autoencoder.py --experiment_name "my_experiment" --loss_type focal_l1
```

**VAE age regression:**
```bash
cd code/project
python 2D_MRI_VAE_regression_torch.py
```

**Cross-modal brain-pathway analysis (requires project/ subdirectory setup):**
```bash
cd code/project/project
python main.py --dataset ACE --experiment_name my_experiment --model NeuroPathX
```

### Data Processing Pipeline

**Step 1: Create 2D transformed images from 3D MRI:**
```bash
cd code/project
python create_2D_transformed_image_I.py   # ABIDE I processing
python create_2D_transformed_image_II.py  # ABIDE II processing
```

**Step 2: Generate synthetic data for augmentation:**
```bash
cd code/project
python create_synthesized_dataset.py
```

### SLURM Job Execution

**Single experiment:**
```bash
sbatch job_scripts/ABIDE_1.sh
```

**Batch processing (multiple cross-validation folds):**
```bash
# Scripts for different CV folds available in job_scripts/scripts/
sbatch job_scripts/scripts/ABIDE_*.sh
```

### Key Training Parameters

The training scripts use extensive hyperparameter configuration via `utils/add_argument.py`. Critical parameters include:

- `--loss_type`: focal_l1, balanced_weighted_l1, simple_l1
- `--experiment_name`: TensorBoard logging identifier
- `--batch_size`: Default 64, adjust based on GPU memory
- `--learning_rate`: Default 0.000005 for cross-modal models
- `--n_epochs`: Default 3000 for convergence
- `--mixed_precision`: Enable FP16 training for memory efficiency

## Important Configuration

### Environment Setup

**Python Environment:**
- Requires Python 3.8.10 with PyTorch 1.13.1
- MONAI virtual environment: `/projectnb/ace-genetics/jueqiw/software/venvs/monai/bin/activate`
- Key dependencies: MONAI, PyTorch, NumPy, pandas, nibabel, SimpleITK

**Path Configuration:**
All data paths are centralized in `code/project/utils/const.py`. Critical paths include:
- ABIDE I/II datasets: `/projectnb/ace-ig/ABIDE/`
- Results output: `/projectnb/ace-ig/jueqiw/experiment/BrainGenePathway/results`
- TensorBoard logs: `/projectnb/ace-ig/jueqiw/experiment/CrossModalityLearning/tensorboard`

### Development Notes

**Configuration Management:**
- `code/project/utils/add_argument.py`: Centralized hyperparameter definitions
- Cross-validation indices are pre-computed and stored as pickle files for reproducibility
- TensorBoard logging controlled via `--not_write_tensorboard` flag
- Thread configuration: `torch.set_num_threads(8)` and `OMP_NUM_THREADS=8` for performance

### Architecture Overview

**Current Active Models:**
- **MONAI AutoEncoder**: Primary model using MONAI's medical imaging framework for 2D brain image processing
- **VAE**: Age regression model using variational autoencoders
- **BrainPathwayAnalysis**: Cross-attention model combining brain imaging with genetic pathway data (legacy, in project/ subdirectory)

**Key Technical Details:**
- Models process 2D slices from 3D MRI volumes (primarily coronal view)
- Input: paired original/transformed brain images (2 channels) → Output: Jacobian deformation maps (1 channel)
- Custom loss functions: focal_l1, balanced_weighted_l1, adaptive_balanced_l1, simple_l1
- Jacobian values normalized to [-1,1] range during training, post-processed back to [0.85,1.15]
- Mixed precision training supported for memory efficiency

### Data Architecture

**Dataset Structure:**
- ABIDE I/II: Autism Brain Imaging Data Exchange datasets
- 2D slices extracted from 3D volumes: axial, coronal, sagittal views
- Data augmentation through synthesized brain images
- Age filtering: subjects ≤21 years old only
- Cross-validation: pre-computed k-fold splits for reproducibility

**Processing Pipeline:**
1. 3D MRI → 2D slice extraction (`create_2D_transformed_image_*.py`)
2. Synthetic data generation (`create_synthesized_dataset.py`)  
3. Training with Jacobian deformation prediction (`weighted_l1_all_monai_autoencoder.py`)

### Compute Environment

**SLURM Integration:**
- Batch job submission via `job_scripts/` with GPU allocation
- Module loading: `python3/3.8.10`, `pytorch/1.13.1`
- Thread optimization: 8 threads for CPU operations
- Memory management: periodic GPU cache clearing during training