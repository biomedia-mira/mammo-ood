from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def load_embed_dataframe(csv_file: str, data_dir: str) -> pd.DataFrame:
    data = pd.read_csv(csv_file, low_memory=False)
    data = data[data["FinalImageType"] == "2D"]
    data = data[data["GENDER_DESC"] == "Female"]
    data = data[data["tissueden"].notna()]
    data = data[data["tissueden"] < 5]
    data = data[data["ViewPosition"].isin(["MLO", "CC"])]
    data = data[data["spot_mag"].isna()]
    data = data.copy().reset_index(drop=True)

    data["img_path"] = [os.path.join(data_dir, str(img_path)) for img_path in data.image_path.values]
    data["study_id"] = [str(study_id) for study_id in data.empi_anon.values]
    data["image_id"] = [str(img_path).split("/")[-1] for img_path in data.image_path.values]
    data["global_index"] = np.arange(len(data))
    # EMBED density labels are encoded as 1..4; make a 0-based target for CE-style losses.
    data["tissueden_index"] = data["tissueden"].astype(int) - 1
    return data


def split_embed_patients(
    data: pd.DataFrame,
    test_percent: float,
    val_percent: float,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    data = data.copy()
    data["split"] = "test"

    unique_study_ids_all = shuffle(data.empi_anon.unique(), random_state=seed)
    num_test = round(len(unique_study_ids_all) * test_percent)

    dev_sub_id = unique_study_ids_all[num_test:]
    data.loc[data.empi_anon.isin(dev_sub_id), "split"] = "training"

    dev_data = data[data["split"] == "training"].copy()
    test_data = data[data["split"] == "test"].copy().reset_index(drop=True)

    unique_study_ids_dev = shuffle(dev_data.empi_anon.unique(), random_state=seed)
    num_train = round(len(unique_study_ids_dev) * (1.0 - val_percent))

    valid_sub_id = unique_study_ids_dev[num_train:]
    dev_data.loc[dev_data.empi_anon.isin(valid_sub_id), "split"] = "validation"

    train_data = dev_data[dev_data["split"] == "training"].copy().reset_index(drop=True)
    val_data = dev_data[dev_data["split"] == "validation"].copy().reset_index(drop=True)

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }


def split_embed_dataframe(
    data: pd.DataFrame,
    test_percent: float,
    val_percent: float,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    return split_embed_patients(
        data,
        test_percent=test_percent,
        val_percent=val_percent,
        seed=seed,
    )
