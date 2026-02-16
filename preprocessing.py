import pandas as pd
import numpy as np
import torch
import os
import glob

# ---------------------------------------------------------
# 1. Dataset Configuration (Label Maps)
# ---------------------------------------------------------
LABEL_MAP_WUT = {
    'Collecting': 0, 'bowing': 1, 'cleaning': 2, 'drinking': 3, 'eating': 4,
    'looking': 5, 'opening': 6, 'passing': 7, 'picking': 8, 'placing': 9,
    'pushing': 10, 'reading': 11, 'sitting': 12, 'standing': 13,
    'standing_up': 14, 'talking': 15, 'turing_front': 16, 'turning': 17, 'walking': 18
}

LABEL_MAP_PKU = {
    "1.0": 0, "3.0": 1, "5.0": 2, "6.0": 3, "7.0": 4,
    "9.0": 5, "11.0": 6, "13.0": 7, "22.0": 8, "25.0": 9,
    "28.0": 10, "32.0": 11, "33.0": 12, "34.0": 13, "35.0": 14,
    "42.0": 15, "44.0": 16, "47.0": 17, "49.0": 18, "51.0": 19
}

DATASET_CONFIGS = {
    'wut': LABEL_MAP_WUT,
    'pku': LABEL_MAP_PKU
}

FEATURE_DIM = 57


# ---------------------------------------------------------
# 2. Internal Logic
# ---------------------------------------------------------

def _read_raw_folder(folder_path):
    """Reads all CSVs in the folder and concatenates them"""
    print(f"   [IO] Scanning folder: {folder_path} ...")
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"Error: No .csv files found under {folder_path}")

    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f, dtype=object)
            df_list.append(df)
        except Exception as e:
            print(f"   Warning: Skipping file {f} ({e})")

    if not df_list: raise ValueError("No data read")

    data = pd.concat(df_list, ignore_index=True, sort=False)

    # Basic cleaning
    data = data.rename(columns=lambda x: x.strip())  # Remove spaces
    data = data.apply(pd.to_numeric, errors='ignore')  # Attempt to convert to numeric

    # Filter out rows with 'not specified'
    if data.columns[-1] in data.columns:
        label_col = data.columns[-1]
        data = data[data[label_col] != "not specified"]

    # Shuffle
    data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return data


def _l2_normalize(df):
    """Vectorized L2 normalization"""
    arr = df.to_numpy().astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return pd.DataFrame(arr / norms, columns=df.columns, index=df.index)


def _preprocess_and_save(raw_folder_path, save_path, label_map):
    """Preprocess and save"""
    df = _read_raw_folder(raw_folder_path)

    features = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
    labels = df.iloc[:, -1]

    if features.isnull().sum().sum() > 0:
        features.fillna(features.mean(), inplace=True)

    print("   [Processing] Performing L2 normalization...")
    features = _l2_normalize(features)

    print(f"   [Processing] Label encoding (Map size: {len(label_map)})...")
    labels_encoded = labels.map(lambda x: label_map.get(str(x), -1))

    valid_mask = (labels_encoded != -1)
    if (~valid_mask).sum() > 0:
        print(f"   [Processing] Removing {(~valid_mask).sum()} rows of invalid label data")
        features = features[valid_mask]
        labels_encoded = labels_encoded[valid_mask]

    # Save
    full_df = pd.concat([features, labels_encoded.rename('label')], axis=1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    full_df.to_csv(save_path, index=False)
    print(f"   [Complete] Saved to: {save_path}")


# ---------------------------------------------------------
# 3. External Interface (Crash issue resolved)
# ---------------------------------------------------------

def get_dataset(raw_path, processed_path, dataset_name='wut', force_update=False):
    """
    Args:
        raw_path: Raw data folder path
        processed_path: Cache file path (.csv)
        dataset_name: 'wut' or 'pku'
        force_update: Whether to force regenerate the cache
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    target_map = DATASET_CONFIGS[dataset_name]

    print(f"\n>>> Loading dataset [{dataset_name.upper()}]")

    # 1. Cache checking logic
    if force_update or not os.path.exists(processed_path):
        print(f"   (Needs to regenerate cache)")
        _preprocess_and_save(raw_path, processed_path, target_map)
    else:
        print(f"   (Reading existing cache: {os.path.basename(processed_path)})")

    # 2. Read CSV
    try:
        df = pd.read_csv(processed_path)
    except Exception as e:
        raise IOError(f"Failed to read CSV: {processed_path}. Error: {e}")

    # 3. Separate features and labels
    X_part = df.iloc[:, :-1]
    y_part = df.iloc[:, -1]

    # -------------------------------------------------------
    # Core fix: Check if the label is a string, if so, map on the fly
    # -------------------------------------------------------
    if y_part.dtype == 'object' or isinstance(y_part.iloc[0], str):
        print(f"   [Warning] The label in the cache file is a string (e.g., '{y_part.iloc[0]}'), mapping on the fly...")

        # Convert using the map corresponding to the passed dataset_name
        y_numeric = y_part.map(lambda x: target_map.get(str(x), -1))

        # Check if conversion was successful
        if (y_numeric == -1).all():
            raise ValueError(f"All label mappings failed! Please check if dataset_name='{dataset_name}' is correct, or if the labels in the CSV are in the MAP.")

        y_part = y_numeric

    # 4. Convert to Tensor
    try:
        X_np = X_part.values.astype(np.float32)
        y_np = y_part.values.astype(np.float32)
    except ValueError as e:
        print(f"   [Fatal Error] Unable to convert data to float: {e}")
        # Print the problematic columns to help with debugging
        print("   Checking for non-numeric characters mixed in feature columns...")
        raise e

    # 5. Dimension correction
    if X_np.shape[1] != FEATURE_DIM:
        try:
            X_np = X_np.reshape(-1, FEATURE_DIM)
        except ValueError:
            pass

    print(f"   Loaded: X={X_np.shape}, y={y_np.shape}")
    return torch.from_numpy(X_np), torch.from_numpy(y_np)
