from argparse import ArgumentParser


def add_argument(parser: ArgumentParser):
    parser.add_argument(
        "--tensor_board_logger",
        default=r"/projectnb/ace-ig/jueqiw/experiment/CrossModalityLearning/tensorboard/",
        help="TensorBoardLogger dir",
    )
    parser.add_argument(
        "--experiment_name",
        default="attention",
        help="Experiment name for TensorBoardLogger",
    )
    parser.add_argument(
        "--normalize_pathway",
        action="store_true",
        help="normalize pathway data",
    )
    parser.add_argument("--test_fold", default=0, type=int)
    parser.add_argument("--run_time", default=0, type=int)
    parser.add_argument("--dataset", choices=["ACE", "ADNI", "SSC"], default="ACE")
    parser.add_argument("--contrastive_loss_weight", default=0, type=float)
    parser.add_argument("--contrastive_margin", default=10, type=float)
    parser.add_argument("--classifier_latent_dim", default=64, type=int)
    parser.add_argument("--learning_rate", default=0.000005, type=float)
    parser.add_argument("--n_epochs", default=300, type=int)
    parser.add_argument(
        "--model",
        default="NeuroPathX",
        choices=["GMIND", "NeuroPathX", "Genetics_Encoder", "UNIGEN"],
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--normalization",
        choices=["batch", "layer", "None"],
        default="batch",
        type=str,
    )
    parser.add_argument(
        "--hidden_dim_qk",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--hidden_dim_k",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--hidden_dim_v",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--soft_sign_constant",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--contrastive_metric",
        choices=["euclidean", "cosine", "L1"],
        default="euclidean",
        type=str,
    )
    parser.add_argument(
        "--pair_loss_weight",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--relu_at_coattention",
        action="store_true",
    )
    parser.add_argument(
        "--hidden_dim_q",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--gene_encoder_layer_2",
        default="10",
        type=str,
    )
    parser.add_argument(
        "--img_encoder_layer_2",
        default="128,64",
        type=str,
    )
    parser.add_argument(
        "--learning_scale",
        default=10,
        type=float,
    )
    parser.add_argument(
        "--img_feature_atlas",
        choices=["Brainnetome", "Destrieux"],
        default="Brainnetome",
        type=str,
    )
    parser.add_argument(
        "--weight_decay",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        # would also start the whole run five folder cross validation
        "--not_write_tensorboard",
        action="store_true",
    )
    parser.add_argument(
        "--n_folder",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--classifier_layer_lr",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--drop_out",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--classifier_drop_out",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--classifier_batch_norm",
        action="store_true",
    )
    parser.add_argument(
        "--img_learnable_drop_out_learning_rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--gene_learnable_drop_out_learning_rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--img_feature_type",
        choices=["thicknessstd", "gauscurv", "thicknessstd_gauscurv"],
        type=str,
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable mixed precision (FP16) training for memory efficiency and speed",
    )
    parser.add_argument(
        "--focal_alpha",
        default=0.25,
        type=float,
        help="Alpha parameter for focal loss (controls overall loss scaling)",
    )
    parser.add_argument(
        "--focal_gamma",
        default=2.0,
        type=float,
        help="Gamma parameter for focal loss (controls focus on hard examples)",
    )
    parser.add_argument(
        "--focal_scale_factor",
        default=10.0,
        type=float,
        help="Scale factor for focal loss to adjust magnitude",
    )
    parser.add_argument(
        "--pos_weight",
        default=None,
        type=float,
        help="Positive class weight for balanced weighted L1 loss (auto-calculated if None)",
    )
    parser.add_argument(
        "--weight_clamp_max",
        default=100.0,
        type=float,
        help="Maximum weight clamp for balanced weighted L1 loss to prevent extreme weights",
    )
    parser.add_argument(
        "--brain_mask_threshold",
        default=0.01,
        type=float,
        help="Threshold for brain mask in balanced weighted L1 loss (areas below this are outside brain)",
    )

    # Learning rate scheduler parameters
    parser.add_argument(
        "--scheduler_type",
        choices=["exponential", "cosine", "step", "plateau"],
        default="exponential",
        type=str,
        help="Type of learning rate scheduler to use",
    )
    parser.add_argument(
        "--gamma",
        default=0.95,
        type=float,
        help="Gamma parameter for scheduler (decay rate for exponential/step, factor for plateau)",
    )
    parser.add_argument(
        "--step_size",
        default=None,
        type=int,
        help="Step size for StepLR scheduler (defaults to epochs//3 if None)",
    )
    parser.add_argument(
        "--patience",
        default=10,
        type=int,
        help="Patience for ReduceLROnPlateau scheduler (number of epochs to wait before reducing LR)",
    )

    # Model architecture parameters
    parser.add_argument(
        "--model_channels",
        default="64,128",
        type=str,
        help="Comma-separated list of channel numbers for each encoder/decoder layer (e.g., '64,128,256')",
    )
    parser.add_argument(
        "--model_strides",
        default="2,2",
        type=str,
        help="Comma-separated list of stride values for each layer (e.g., '2,2,2')",
    )
    parser.add_argument(
        "--num_res_units",
        default=3,
        type=int,
        help="Number of residual units in each layer",
    )
    parser.add_argument(
        "--model_dropout",
        default=0.05,
        type=float,
        help="Dropout rate for the model",
    )

    # VAE training parameters
    parser.add_argument(
        "--stage_transition_epoch",
        default=40,
        type=int,
        help="Epoch at which to transition from warm-up stage to fine-tuning stage in VAE training",
    )
    parser.add_argument(
        "--latent_dim",
        default=32,
        type=int,
        help="Latent dimension for VAE",
    )
    parser.add_argument(
        "--recon_weight",
        default=1.0,
        type=float,
        help="Weight for reconstruction loss in VAE",
    )
    parser.add_argument(
        "--kl_weight",
        default=0.01,
        type=float,
        help="Weight for KL divergence loss in VAE",
    )
    parser.add_argument(
        "--age_weight",
        default=10.0,
        type=float,
        help="Weight for age regression loss in VAE",
    )

    # GAN-specific arguments
    parser.add_argument(
        "--adversarial_weight",
        default=1.0,
        type=float,
        help="Weight for adversarial loss in GAN training",
    )
    parser.add_argument(
        "--content_weight",
        default=10.0,
        type=float,
        help="Weight for content/reconstruction loss in GAN training",
    )
    parser.add_argument(
        "--gradient_penalty_weight",
        default=10.0,
        type=float,
        help="Weight for gradient penalty in WGAN-GP",
    )
    parser.add_argument(
        "--use_wgan_gp",
        action="store_true",
        help="Use WGAN-GP instead of standard GAN loss",
    )
    parser.add_argument(
        "--lr_generator",
        default=2e-4,
        type=float,
        help="Learning rate for GAN generator",
    )
    parser.add_argument(
        "--lr_discriminator",
        default=2e-4,
        type=float,
        help="Learning rate for GAN discriminator",
    )
    parser.add_argument(
        "--filters",
        default=32,  # Reduced from 64 for memory efficiency
        type=int,
        help="Number of base filters for GAN models",
    )
    parser.add_argument(
        "--latent_space",
        default=64,  # Reduced from 128 for memory efficiency
        type=int,
        help="Dimension of latent space for GAN generator",
    )
    parser.add_argument(
        "--perceptual_weight",
        default=0.1,
        type=float,
        help="Weight for perceptual loss in GAN training",
    )
    parser.add_argument(
        "--d_train_freq",
        default=1,
        type=int,
        help="Frequency for training discriminator (1=every iteration, 2=every 2nd iteration)",
    )
    parser.add_argument(
        "--g_train_freq",
        default=1,
        type=int,
        help="Frequency for training generator (1=every iteration, 2=every 2nd iteration)",
    )
