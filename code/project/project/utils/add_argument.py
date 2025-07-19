from argparse import ArgumentParser


def add_argument(parser: ArgumentParser):
    parser.add_argument(
        "--tensor_board_logger",
        default=r"/projectnb/ace-ig/jueqiw/experiment/CrossModalityLearning/tensorboard/",
        help="TensorBoardLogger dir",
    )
    parser.add_argument(
        "--experiment_name",
        default="attention_5",
        help="Experiment name for TensorBoardLogger",
    )
    parser.add_argument(
        "--normalize_pathway",
        action="store_true",
        help="normalize pathway data",
    )
    parser.add_argument(
        "--diff_pair_loss",
        action="store_true",
    )
    parser.add_argument(
        "--top_k",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--sparsity_loss_weight",
        default=1e-4,
        type=float,
    )
    parser.add_argument(
        "--bernoulli_probability",
        default=1e-5,
        type=float,
    )
    parser.add_argument("--test_fold", default=0, type=int)
    parser.add_argument("--run_time", default=0, type=int)
    parser.add_argument("--dataset", choices=["ACE", "ADNI", "SSC"], default="ACE")
    parser.add_argument("--contrastive_loss_weight", default=0, type=float)
    parser.add_argument("--contrastive_margin", default=10, type=float)
    parser.add_argument("--classifier_latent_dim", default=64, type=int)
    parser.add_argument("--learning_rate", default=0.000005, type=float)
    parser.add_argument("--n_epochs", default=3000, type=int)
    parser.add_argument(
        "--model",
        default="NeuroPathX",
        choices=["GMIND", "NeuroPathX", "Genetics_Encoder"],
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
        "--only_BCE_loss",
        action="store_true",
    )
    parser.add_argument(
        "--classification_loss_type",
        choices=["BCE", "Ordinal"],
        default="BCE",
        type=str,
    )
    parser.add_argument(
        "--class_loss_weight",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--img_recon_loss_weight",
        default=0.0001,
        type=float,
    )
    parser.add_argument(
        "--gene_recon_loss_weight",
        default=0.001,
        type=float,
    )
    parser.add_argument(
        "--sparse_loss_weight_img",
        default=0.001,
        type=float,
    )
    parser.add_argument(
        "--sparse_loss_weight_gene",
        default=0.001,
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
