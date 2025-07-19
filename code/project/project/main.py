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

# from code.project.models.model import BrainPathwayAnalysis
from models.losses import max_margin_contrastive_loss, generate_pairs, pattern_loss
from utils.utils import (
    seed_everything,
    normalize_data,
    add_log,
    save_result_dataframe,
    preprocess_df_ACE,
    preprocess_df_AD,
)
from utils.data_loader import BrainPathwayDataset
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
            img, pathway, label = (
                data["img"].to(device).float(),
                data["pathway"].to(device).float(),
                data["label"].to(device).float(),
            )

            if len(torch.unique(label)) == 1:
                continue
            else:
                optimizer.zero_grad()
                output, attn_weights, z_vector = model(img, pathway)
                bce_loss = criterion(output, label)
                if img.shape[0] >= 20 and hparams.diff_pair_loss:
                    # 10 pairs from the same group and 10 pairs from different groups
                    group_pairs = generate_pairs(label, num_pairs=20)
                    pair_loss = pattern_loss(
                        attn_weights, group_pairs, metric=hparams.contrastive_metric
                    )
                    pair_loss = pair_loss / (len(group_pairs) // 2)
                    total_loss = bce_loss + pair_loss * hparams.pair_loss_weight
                else:
                    total_loss = bce_loss

                if img.shape[0] >= 20 and hparams.sparsity_loss_weight > 0:
                    sparsity_loss = torch.tensor(0.0).to(device)
                    # random choose 10 samples from the batch
                    samples = np.random.choice(img.shape[0], 2, replace=False)
                    attn_weights_flatten = attn_weights[samples].flatten()

                    rho = torch.FloatTensor(
                        [
                            bernoulli_probability
                            for _ in range(attn_weights_flatten.size()[0])
                        ]
                    ).to(device)
                    rho_hat = attn_weights_flatten

                    # KL divergence
                    x1 = rho
                    x2 = rho_hat
                    # ensure the probability is sparse
                    s1 = torch.sum(x2 * (torch.log(x2 + eps) - torch.log(x1 + eps)))
                    s2 = (
                        torch.sum(
                            (1 - x2)
                            * (torch.log(1 - x2 + eps) - torch.log(1 - x1 + eps))
                        )
                        + s1
                    )
                    sparsity_loss = s2 / 2
                    total_loss += sparsity_loss * hparams.sparsity_loss_weight
                    sparsity_losses.append(
                        sparsity_loss.item() * hparams.sparsity_loss_weight
                    )
                total_loss.backward(retain_graph=True)
                optimizer.step()
                bce_losses.append(bce_loss.item())
                total_losses.append(total_loss.item())
                different_pair_loss.append(pair_loss.item() * hparams.pair_loss_weight)
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
            img, pathway, label = (
                data["img"].to(device).float(),
                data["pathway"].to(device).float(),
                data["label"].to(device).float(),
            )
            output, attn_weights, z_vector = model(img, pathway)
            bce_loss = criterion(output, label)
            group_pairs = generate_pairs(label, num_pairs=8)
            pair_loss = pattern_loss(
                attn_weights, group_pairs, metric=hparams.contrastive_metric
            )
            total_loss = torch.tensor(0.0).to(device)
            if hparams.diff_pair_loss:
                pair_loss = pair_loss / (len(group_pairs) // 2)
                total_loss = pair_loss * hparams.pair_loss_weight
            total_loss += bce_loss

            sparsity_loss = torch.tensor(0.0).to(device)
            if hparams.sparsity_loss_weight > 0:
                # random choose 10 samples from the batch
                samples = np.random.choice(img.shape[0], 2, replace=False)
                attn_weights_flatten = attn_weights[samples].flatten()

                rho = torch.FloatTensor(
                    [
                        bernoulli_probability
                        for _ in range(attn_weights_flatten.size()[0])
                    ]
                ).to(device)
                rho_hat = attn_weights_flatten

                # KL divergence
                x1 = rho
                x2 = rho_hat
                # ensure the probability is sparse
                s1 = torch.sum(x2 * (torch.log(x2 + eps) - torch.log(x1 + eps)))
                s2 = (
                    torch.sum(
                        (1 - x2) * (torch.log(1 - x2 + eps) - torch.log(1 - x1 + eps))
                    )
                    + s1
                )
                sparsity_loss = s2 / 2
                total_loss += sparsity_loss * hparams.sparsity_loss_weight
                val_sparsity_losses.append(
                    sparsity_loss.item() * hparams.sparsity_loss_weight
                )

            val_bce_losses.append(bce_loss.item())
            if hparams.diff_pair_loss:
                val_different_pair_loss.append(
                    pair_loss.item() * hparams.pair_loss_weight
                )
            val_total_losses.append(total_loss.item())
            val_outputs.append(output.detach().cpu())
            val_labels.append(label.detach().cpu().numpy())

            if epoch % 100 == 0 and not hparams.not_write_tensorboard:
                # random number from 0 - 31
                sample_idx = np.random.randint(0, attn_weights.shape[0])
                label_np = label[sample_idx].detach().cpu().numpy()
                attention_matrix = attn_weights[sample_idx].detach().cpu().numpy()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(attention_matrix, cmap="viridis", ax=ax, cbar=True)
                ax.set_xlabel("Key Features (e.g., Image Features)")
                ax.set_ylabel("Query Features (e.g., Genetic Pathway Features)")
                title = f"Attention Matrix at Epoch {epoch} for Sample {sample_idx} label {label_np}"
                writer.add_figure(tag=title, figure=fig, global_step=epoch)
                plt.close(fig)

                # log the z_vector
                z_vector_np = z_vector.detach().cpu().numpy().squeeze()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(z_vector_np, cmap="viridis", ax=ax, cbar=True)
                # change the np array to string
                ax.set_xticks(np.arange(z_vector_np.shape[1]))
                ax.set_xticklabels(np.arange(z_vector_np.shape[1]))
                ax.set_xlabel("Z Vector Dimension")
                ax.set_ylabel("Batch Dimension")
                title = f"Z Vector at Epoch {epoch}"
                writer.add_figure(tag=title, figure=fig, global_step=epoch)
                plt.close(fig)

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
            if hparams.diff_pair_loss:
                diff_pair_loss = np.mean(different_pair_loss)
                val_diff_pair_loss = np.mean(val_different_pair_loss)
                writer.add_scalar("Loss/train_diff_pair", diff_pair_loss, epoch)
                writer.add_scalar("Loss/val_diff_pair", val_diff_pair_loss, epoch)
            if hparams.sparsity_loss_weight > 0:
                sparsity_loss = np.mean(sparsity_losses)
                val_sparsity_loss = np.mean(val_sparsity_losses)
                writer.add_scalar("Loss/train_sparsity", sparsity_loss, epoch)
                writer.add_scalar("Loss/val_sparsity", val_sparsity_loss, epoch)
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
        img, pathway, label = (
            data["img"].to(device).float(),
            data["pathway"].to(device).float(),
            data["label"].to(device).float(),
        )
        output, attn_weights, z_vector = model(img, pathway)
        test_outputs.append(output.detach().cpu())
        test_labels.append(label.detach().cpu().numpy())

        if epoch == 1300:
            # save the attention matrix
            for i in range(attn_weights.shape[0]):
                attention_matrix = attn_weights[i].detach().cpu().numpy()
                sample_label = label[i].detach().cpu().numpy()
                asd_folder = result_fold_path / "ASD"
                con_folder = result_fold_path / "CON"
                if not os.path.exists(asd_folder):
                    os.makedirs(asd_folder)
                if not os.path.exists(con_folder):
                    os.makedirs(con_folder)
                if sample_label == 1:
                    np.save(
                        asd_folder
                        / f"attention_matrix_sample_{i}_epoch_{epoch}_fold_{n_fold}_time_{run_time}.npy",
                        attention_matrix,
                    )
                else:
                    np.save(
                        con_folder
                        / f"attention_matrix_sample_{i}_epoch_{epoch}_fold_{n_fold}_time_{run_time}.npy",
                        attention_matrix,
                    )

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
    if hparams.dataset == "ACE":
        img_pathway = pd.read_csv(ACE_FILE)
        img_pathway = img_pathway.drop(columns=img_pathway.columns[0])
        img, pathway, label, father_site_ids, new_ids = preprocess_df_ACE(img_pathway)
        # img shape (200, 840)
        cross_val_index = CROSS_VAL_INDEX_ACE
    elif hparams.dataset == "ADNI":
        img_pathway = pd.read_csv(ADNI_FILE)
        img_pathway = img_pathway.drop(columns=img_pathway.columns[0])
        img, pathway, label = preprocess_df_AD(img_pathway)
        cross_val_index = CROSS_VAL_INDEX_ADNI

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
            X_train_img,
            X_train_pathway,
            y_train,
            X_val_img,
            X_val_pathway,
            y_val,
            X_test_img,
            X_test_pathway,
            y_test,
        ) = ten_fold_cross_validation(
            img,
            pathway,
            label,
            test_fold=test_fold,
            run_time=hparams.run_time,
            cross_val_index=cross_val_index,
            hparams=hparams,
            father_site_ids=father_site_ids,
            new_ids=new_ids,
        )

        X_train_img, X_val_img, X_test_img = normalize_data(
            X_train_img, X_val_img, X_test_img
        )

        if hparams.normalize_pathway:
            X_train_pathway, X_val_pathway, X_test_pathway = normalize_data(
                X_train_pathway, X_val_pathway, X_test_pathway
            )

        # load the data to the dataloader
        train_dataset = BrainPathwayDataset(
            X_train_img, X_train_pathway, y_train, hparams=hparams
        )
        val_dataset = BrainPathwayDataset(
            X_val_img, X_val_pathway, y_val, hparams=hparams
        )
        test_dataset = BrainPathwayDataset(
            X_test_img, X_test_pathway, y_test, hparams=hparams
        )

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
