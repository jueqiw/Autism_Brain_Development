import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from utils.utils import preprocess_df_ACE, seed_everything, preprocess_df_AD

if __name__ == "__main__":
    img_pathway = pd.read_csv(
        "/projectnb/ace-ig/jueqiw/dataset/BrainGenePathway/ADNI/Gene/final_AD_KEGG_pathway_with_all_genes_img_p_threshold_0.1_effect_size_LD_50kb.csv"
    )
    # drop the first column
    img_pathway = img_pathway.drop(columns=img_pathway.columns[0])
    img, pathway, label = preprocess_df_AD(img_pathway)

    save_dict = {}
    seed_everything(50)
    for time in range(10):
        for test_fold in range(10):
            skf = StratifiedKFold(n_splits=10, shuffle=True)
            folders = []

            splits = skf.split(pathway, label)
            for i, (_, test_index) in enumerate(splits):
                folders.append(test_index)

            val_fold = (test_fold + 1) % 10
            train_fold = set(i for i in range(10)) - set([test_fold, val_fold])
            train_index = np.concatenate([folders[i] for i in train_fold])

            print(f"time_{time}_fold_{test_fold}")
            print("val_index", folders[val_fold])
            print("test_index", folders[test_fold])

            save_dict[f"time_{time}_fold_{test_fold}_val"] = folders[val_fold]
            save_dict[f"time_{time}_fold_{test_fold}_test"] = folders[test_fold]

    # dump the save_dict
    with open(
        "/projectnb/ace-ig/jueqiw/dataset/BrainGenePathway/ADNI/10_10_cross_fold_val_index.pkl",
        "wb",
    ) as f:
        pickle.dump(save_dict, f)
