import pandas as pd
import torch

from utils.const import DATA_FOLDER, TADPOLE_FOLDER


def choose_column():
    column_meaing_csv = pd.read_csv(TADPOLE_FOLDER / "TADPOLE_D1_D2_Dict.csv")
    # choose part of the table by the third column
    part_table = column_meaing_csv[column_meaing_csv["TBLNAME"] == "UCSFFSX"]
    # print total number of part_table which the 6 columns contain "Cortical Thickness"
    chosen_column = part_table[
        part_table["TEXT"].str.contains("Cortical Thickness Average", na=False)
    ]
    # print the "TEXT" column of chosen_column without omitting anything
    chosen_column_list = []
    for i in range(len(chosen_column)):
        chosen_column_list.append(chosen_column.iloc[i]["FLDNAME"])
    return chosen_column_list


def generate_image_feature_adni_go():
    tadpole_go2 = pd.read_csv(TADPOLE_FOLDER / "tadpole_go.csv")
    chosen_column_list = choose_column()
    print(len(chosen_column_list))

    filtered_columns = [
        col
        for col in tadpole_go2.columns
        if col in chosen_column_list
        or ("PTID" in col)
        or ("DX" in col)
        or ("DX_bl" in col)
    ]
    tadpole_go2 = tadpole_go2[filtered_columns]

    # create new column based on DX_bl
    tadpole_go2["class_3"] = tadpole_go2["DX_bl"].replace(
        {"CN": 0, "SMC": 0, "EMCI": 1, "LMCI": 1, "AD": 2}
    )
    tadpole_go2["class_4"] = tadpole_go2["DX_bl"].replace(
        {"CN": 0, "SMC": 0, "EMCI": 1, "LMCI": 2, "AD": 3}
    )

    # drop column
    drop_columns = [
        "INDGLBNDX_BAIPETNMRC_09_12_16",
        "INDTEPNDX_BAIPETNMRC_09_12_16",
        "DXCHANGE",
        "DX",
        "ST123TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        "ST22TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        "ST64TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
        "ST81TA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16",
    ]
    tadpole_go2 = tadpole_go2.drop(drop_columns, axis=1)
    tadpole_go2 = tadpole_go2.dropna()
    tadpole_go2.to_csv(TADPOLE_FOLDER / "tadpole_go_feature.csv", index=False)
    return tadpole_go2


def cpu(x):
    x = x.to(torch.device("cpu"))
    return x


def cpu_t(x):
    y = torch.tensor(x.astype(float)).float()
    return y


def cpu_ts(x):
    y = torch.tensor(x).float()
    return y


def gpu(x):
    x = x.to("cuda")
    return x


def gpu_t(x):
    x = torch.tensor(x.astype(float)).float()
    y = x.to("cuda")
    return y


def gpu_ts(x):
    x = torch.tensor(x).float()
    y = x.to("cuda")
    return y
