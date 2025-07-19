from typing import Tuple
from argparse import ArgumentParser, Namespace
from pathlib import Path
import os
import sys
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from project.models.genetics_encoder import BrainPathwayAnalysis, PathwayEncoder
from models.losses import max_margin_contrastive_loss, generate_pairs, pattern_loss
from utils.utils import (
    seed_everything,
    normalize_data,
    add_log,
    save_result_dataframe,
    preprocess_df_ACE,
    preprocess_df_AD,
    preprocess_df_SSC,
)
from utils.data_loader import BrainPathwayDataset, PathwayEncoderDataset
from utils.add_argument import add_argument
from utils.const import (
    RESULT_FOLDER,
    ACE_FILE,
    ADNI_FILE,
    CROSS_VAL_INDEX_ACE,
    CROSS_VAL_INDEX_ADNI,
)

torch.autograd.set_detect_anomaly(True)


def create_folder(hparams: Namespace) -> Path:
    result_fold_path = (
        RESULT_FOLDER / f"{hparams.dataset}" / f"{hparams.experiment_name}"
    )
    if not os.path.exists(result_fold_path):
        os.makedirs(result_fold_path)
        print(f"Create folder: {result_fold_path}!")

    return result_fold_path


def ten_fold_cross_validation_SSC(
    pathway: pd.DataFrame,
    label: pd.DataFrame,
    cross_val_index: Path = None,
    test_fold: int = 0,
    run_time: int = 0,
    hparams: Namespace = None,
):
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    folders = []

    splits = skf.split(pathway, label)
    for i, (_, test_index) in enumerate(splits):
        folders.append(test_index)

    val_fold_id = (test_fold + 1) % 10
    train_fold = set(i for i in range(10)) - set([test_fold, val_fold_id])
    train_index = np.concatenate([folders[i] for i in train_fold])
    print("test_fold", test_fold)
    print("val_fold", val_fold_id)
    print("train_fold", train_fold)

    X_train_pathway, y_train = (
        pathway.iloc[train_index],
        label.iloc[train_index],
    )
    X_val_pathway, y_val = (
        pathway.iloc[folders[val_fold_id]],
        label.iloc[folders[val_fold_id]],
    )
    X_test_pathway, y_test = (
        pathway.iloc[folders[test_fold]],
        label.iloc[folders[test_fold]],
    )

    return (
        X_train_pathway,
        y_train,
        X_val_pathway,
        y_val,
        X_test_pathway,
        y_test,
    )


def ten_fold_cross_validation(
    img: pd.DataFrame,
    pathway: pd.DataFrame,
    label: pd.DataFrame,
    cross_val_index: Path,
    test_fold: int = 0,
    run_time: int = 0,
    hparams: Namespace = None,
    father_site_ids: pd.DataFrame = None,
    new_ids: pd.DataFrame = None,
):
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    folders = []

    # load pickle file
    with open(cross_val_index, "rb") as f:
        indx = pickle.load(f)

    if hparams.dataset == "ACE":
        # First Remove the related samples
        relatedness_csv = pd.read_csv(
            "/projectnb/ace-ig/jueqiw/dataset/SSC_Pseudo_ACE_onekg_call_0.2/ACE/remove_related_samples_new.txt",
            sep=" ",
            header=None,
        )
        # for each related sample, find the corresponding father site id
        related_father_site_ids = father_site_ids[new_ids.isin(relatedness_csv[1])]
        print("The number of related samples is: ", len(related_father_site_ids))  #
        print(
            "The number of unique father site ids is: ",
            len(related_father_site_ids.unique()),
        )
        print(
            "# of relatedness ids in the new ids: ",
            len(relatedness_csv[1]),
        )

        relatedness_csv = new_ids[
            new_ids.isin(related_father_site_ids.unique())
        ].reset_index(drop=True)

        splits = skf.split(pathway, label)
        for i, (_, test_index) in enumerate(splits):
            folders.append(test_index)
    else:
        splits = skf.split(pathway, label)
        for i, (_, test_index) in enumerate(splits):
            folders.append(test_index)

    test_index = indx[f"time_{run_time}_fold_{test_fold}_test"]
    val_index = indx[f"time_{run_time}_fold_{test_fold}_val"]
    all_index = [i for i in range(len(label))]
    train_index = list(set(all_index) - set(test_index) - set(val_index))
    print("test_fold", test_index)
    print("val_fold", val_index)

    X_train_img, X_train_pathway, y_train = (
        img.iloc[train_index],
        pathway.iloc[train_index],
        label.iloc[train_index],
    )
    X_val_img, X_val_pathway, y_val = (
        img.iloc[val_index],
        pathway.iloc[val_index],
        label.iloc[val_index],
    )
    X_test_img, X_test_pathway, y_test = (
        img.iloc[test_index],
        pathway.iloc[test_index],
        label.iloc[test_index],
    )

    return (
        X_train_img.copy(),
        X_train_pathway.copy(),
        y_train.copy(),
        X_val_img.copy(),
        X_val_pathway.copy(),
        y_val.copy(),
        X_test_img.copy(),
        X_test_pathway.copy(),
        y_test.copy(),
    )


def train_model(
    train_loader,
    val_loader,
    test_loader,
    model,
    optimizer,
    writer,
    device,
    criterion,
    result_fold_path: Path,
    hparams: Namespace,
    n_fold: int,
    result_dict: dict,
    n_epochs: int = 3000,
    ACE_pathway: pd.DataFrame = None,
    ACE_label: pd.DataFrame = None,
):
    for epoch in range(n_epochs):
        model.train()
        model.device = device
        (
            bce_losses,
            different_pair_loss,
            total_losses,
            sparsity_losses,
        ) = ([], [], [], [])
        total_loss = torch.tensor(0.0).to(device)
        bernoulli_probability = torch.tensor(hparams.bernoulli_probability).to(device)
        eps = torch.tensor(1e-10).to(device)
        for i, data in enumerate(train_loader):
            pathway, label = (
                data["pathway"].to(device).float(),
                data["label"].to(device).float(),
            )

            if len(torch.unique(label)) == 1:
                continue
            else:
                optimizer.zero_grad()
                output = model(pathway)
                bce_loss = criterion(output, label)
                total_loss = bce_loss

                total_loss.backward(retain_graph=True)
                optimizer.step()
                bce_losses.append(bce_loss.item())
                total_losses.append(total_loss.item())
                add_log(
                    model="train",
                    y_label=label.detach().cpu().numpy(),
                    output=output.detach(),
                    hparams=hparams,
                    writer=writer,
                    epoch=epoch,
                    result_dict=result_dict,
                    fold=n_fold,
                )

        model.eval()
        (
            val_bce_losses,
            val_different_pair_loss,
            val_sparsity_losses,
            val_total_losses,
            val_outputs,
            val_labels,
        ) = ([], [], [], [], [], [])
        for i, data in enumerate(val_loader):
            pathway, label = (
                data["pathway"].to(device).float(),
                data["label"].to(device).float(),
            )
            output = model(pathway)
            bce_loss = criterion(output, label)
            total_loss = torch.tensor(0.0).to(device)
            total_loss += bce_loss

            sparsity_loss = torch.tensor(0.0).to(device)

            val_bce_losses.append(bce_loss.item())
            val_total_losses.append(total_loss.item())
            val_outputs.append(output.detach().cpu())
            val_labels.append(label.detach().cpu().numpy())

        output = torch.cat(val_outputs)
        label = np.concatenate(val_labels)
        add_log(
            model="val",
            y_label=label,
            output=output,
            hparams=hparams,
            writer=writer,
            epoch=epoch,
            result_dict=result_dict,
            fold=n_fold,
        )

        if not hparams.not_write_tensorboard:
            bce_loss = np.mean(bce_losses)
            val_bce_loss = np.mean(val_bce_losses)
            writer.add_scalar("Loss/train_bce", bce_loss, epoch)
            writer.add_scalar("Loss/train_total_loss", np.mean(total_losses), epoch)
            writer.add_scalar("Loss/val_bce", val_bce_loss, epoch)
            writer.add_scalar("Loss/val_total_loss", np.mean(val_total_losses), epoch)

        test(
            model,
            test_loader,
            device,
            criterion,
            writer,
            epoch,
            result_dict,
            n_fold,
            run_time=hparams.run_time,
            result_fold_path=result_fold_path,
        )

    return model


def test(
    model,
    test_loader,
    device,
    criterion,
    writer,
    epoch,
    result_dict,
    n_fold,
    run_time,
    result_fold_path,
):
    model.eval()
    test_outputs, test_labels = [], []

    for i, data in enumerate(test_loader):
        pathway, label = (
            data["pathway"].to(device).float(),
            data["label"].to(device).float(),
        )
        output = model(pathway)
        test_outputs.append(output.detach().cpu())
        test_labels.append(label.detach().cpu().numpy())

    output = torch.cat(test_outputs)
    label = np.concatenate(test_labels)
    add_log(
        model="test",
        y_label=label,
        output=output,
        hparams=hparams,
        writer=writer,
        epoch=epoch,
        result_dict=result_dict,
        fold=n_fold,
    )


def main(hparams: Namespace):
    father_site_ids, new_ids = None, None
    if hparams.dataset == "SSC":
        pathway_data = pd.read_csv(
            "/projectnb/ace-ig/jueqiw/dataset/CrossModalityLearning/SSC/CSV/final_SSC_pseudo_KEGG_pathway_with_all_genes_p_threshold_0.1_effect_size_LD_50kb.csv"
        )
        pathway_data = pathway_data.drop(columns=pathway_data.columns[0])
        pathway, label = preprocess_df_SSC(pathway_data)
    elif hparams.dataset == "ADNI":
        img_pathway = pd.read_csv(ADNI_FILE)
        img_pathway = img_pathway.drop(columns=img_pathway.columns[0])
        img, pathway, label = preprocess_df_AD(img_pathway)
        cross_val_index = CROSS_VAL_INDEX_ADNI

    # load test dataset
    img_pathway = pd.read_csv(ACE_FILE)
    img_pathway = img_pathway.drop(columns=img_pathway.columns[0])
    ACE_img, ACE_pathway, ACE_label, ACE_father_site_ids, ACE_new_ids = (
        preprocess_df_ACE(img_pathway)
    )
    # img shape (200, 840)
    cross_val_index = CROSS_VAL_INDEX_ACE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dict = {
        "fold": [],
        "mode": [],
        "Epoch": [],
        "Acc": [],
        "F1": [],
        "Kappa": [],
        "Sensitivity": [],
        "Specificity": [],
        "AUC": [],
    }
    result_fold_path = create_folder(hparams)

    for test_fold in range(10):
        # if not hparams.not_write_tensorboard:
        if test_fold != hparams.test_fold:
            continue

        seed_everything(50)
        (
            X_train_pathway,
            y_train,
            X_val_pathway,
            y_val,
            X_test_pathway,
            y_test,
        ) = ten_fold_cross_validation_SSC(
            pathway=pathway,
            label=label,
            test_fold=test_fold,
            hparams=hparams,
        )

        if hparams.normalize_pathway:
            X_train_pathway, X_val_pathway, X_test_pathway = normalize_data(
                X_train_pathway, X_val_pathway, X_test_pathway
            )

        # load the data to the dataloader
        train_dataset = PathwayEncoderDataset(X_train_pathway, y_train, hparams=hparams)
        val_dataset = PathwayEncoderDataset(X_val_pathway, y_val, hparams=hparams)
        test_dataset = PathwayEncoderDataset(ACE_pathway, ACE_label, hparams=hparams)

        train_loader = DataLoader(
            train_dataset, batch_size=hparams.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=hparams.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=hparams.batch_size, shuffle=False
        )

        writer = None
        if not hparams.not_write_tensorboard:
            writer = SummaryWriter(
                log_dir=Path(hparams.tensor_board_logger) / hparams.experiment_name
            )

        if hparams.model == "Genetics_Encoder":
            model = PathwayEncoder(
                n_pathway=X_train_pathway.shape[1],
                classifier_latnet_dim=hparams.classifier_latent_dim,
                normalization=hparams.normalization,
                hidden_dim_qk=hparams.hidden_dim_qk,
                hidden_dim_q=hparams.hidden_dim_q,
                hidden_dim_k=hparams.hidden_dim_k,
                hidden_dim_v=hparams.hidden_dim_v,
                relu_at_coattention=hparams.relu_at_coattention,
                soft_sign_constant=hparams.soft_sign_constant,
            ).to(device)

        if hparams.model == "NeuroPathX":
            model = BrainPathwayAnalysis(
                n_img_features=X_train_img.shape[1] // 4,
                n_pathway=X_train_pathway.shape[1],
                classifier_latnet_dim=hparams.classifier_latent_dim,
                normalization=hparams.normalization,
                hidden_dim_qk=hparams.hidden_dim_qk,
                hidden_dim_q=hparams.hidden_dim_q,
                hidden_dim_k=hparams.hidden_dim_k,
                hidden_dim_v=hparams.hidden_dim_v,
                relu_at_coattention=hparams.relu_at_coattention,
                soft_sign_constant=hparams.soft_sign_constant,
            ).to(device)

        n_ASD, n_CON = y_train.sum(), len(y_train) - y_train.sum()
        pos_weight = torch.tensor([n_CON / n_ASD], device=device)
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
        train_model(
            train_loader,
            val_loader,
            test_loader,
            model,
            n_fold=test_fold,
            optimizer=torch.optim.AdamW(
                model.parameters(), lr=hparams.learning_rate, weight_decay=1e-5
            ),
            writer=writer,
            device=device,
            result_fold_path=result_fold_path,
            criterion=loss,
            hparams=hparams,
            result_dict=result_dict,
            n_epochs=hparams.n_epochs,
            ACE_pathway=ACE_pathway,
            ACE_label=ACE_label,
        )

    if hparams.not_write_tensorboard:
        save_result_dataframe(
            result_fold_path=result_fold_path,
            df_result=result_dict,
            hparams=hparams,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Trainer args", add_help=False)
    add_argument(parser)
    hparams = parser.parse_args()
    main(hparams)
