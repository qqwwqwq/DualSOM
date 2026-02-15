
import pandas as pd
import numpy as np
import torch
import os
import glob

# ---------------------------------------------------------
# 1. 数据集配置 (Label Maps)
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
# 2. 内部逻辑
# ---------------------------------------------------------

def _read_raw_folder(folder_path):
    """读取文件夹下所有 CSV 并合并"""
    print(f"   [IO]正在扫描文件夹: {folder_path} ...")
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"错误: 在 {folder_path} 下未找到任何 .csv 文件")

    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f, dtype=object)
            df_list.append(df)
        except Exception as e:
            print(f"   警告: 跳过文件 {f} ({e})")

    if not df_list: raise ValueError("没有读取到数据")

    data = pd.concat(df_list, ignore_index=True, sort=False)

    # 基础清洗
    data = data.rename(columns=lambda x: x.strip())  # 去空格
    data = data.apply(pd.to_numeric, errors='ignore')  # 尝试转数值

    # 过滤掉 'not specified' 的行
    if data.columns[-1] in data.columns:
        label_col = data.columns[-1]
        data = data[data[label_col] != "not specified"]

    # 打乱
    data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return data


def _l2_normalize(df):
    """向量化 L2 归一化"""
    arr = df.to_numpy().astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return pd.DataFrame(arr / norms, columns=df.columns, index=df.index)


def _preprocess_and_save(raw_folder_path, save_path, label_map):
    """预处理并保存"""
    df = _read_raw_folder(raw_folder_path)

    features = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
    labels = df.iloc[:, -1]

    if features.isnull().sum().sum() > 0:
        features.fillna(features.mean(), inplace=True)

    print("   [处理] 执行 L2 归一化...")
    features = _l2_normalize(features)

    print(f"   [处理] 标签编码 (Map size: {len(label_map)})...")
    labels_encoded = labels.map(lambda x: label_map.get(str(x), -1))

    valid_mask = (labels_encoded != -1)
    if (~valid_mask).sum() > 0:
        print(f"   [处理] 剔除 {(~valid_mask).sum()} 行无效标签数据")
        features = features[valid_mask]
        labels_encoded = labels_encoded[valid_mask]

    # 保存
    full_df = pd.concat([features, labels_encoded.rename('label')], axis=1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    full_df.to_csv(save_path, index=False)
    print(f"   [完成] 已保存至: {save_path}")


# ---------------------------------------------------------
# 3. 对外接口 (已修复崩溃问题)
# ---------------------------------------------------------

def get_dataset(raw_path, processed_path, dataset_name='wut', force_update=False):
    """
    Args:
        raw_path: 原始数据文件夹路径
        processed_path: 缓存文件路径 (.csv)
        dataset_name: 'wut' 或 'pku'
        force_update: 是否强制重新生成缓存
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    target_map = DATASET_CONFIGS[dataset_name]

    print(f"\n>>> 正在加载数据集 [{dataset_name.upper()}]")

    # 1. 缓存检查逻辑
    if force_update or not os.path.exists(processed_path):
        print(f"   (需要重新生成缓存)")
        _preprocess_and_save(raw_path, processed_path, target_map)
    else:
        print(f"   (读取现有缓存: {os.path.basename(processed_path)})")

    # 2. 读取 CSV
    try:
        df = pd.read_csv(processed_path)
    except Exception as e:
        raise IOError(f"读取 CSV 失败: {processed_path}. 错误: {e}")

    # 3. 分离特征和标签
    X_part = df.iloc[:, :-1]
    y_part = df.iloc[:, -1]

    # -------------------------------------------------------
    # 核心修复: 检查标签是否为字符串，如果是，现场转换
    # -------------------------------------------------------
    if y_part.dtype == 'object' or isinstance(y_part.iloc[0], str):
        print(f"   [警告] 缓存文件中的标签是字符串 (例如 '{y_part.iloc[0]}')，正在进行实时映射...")

        # 使用传入的 dataset_name 对应的 map 进行转换
        y_numeric = y_part.map(lambda x: target_map.get(str(x), -1))

        # 检查是否转换成功
        if (y_numeric == -1).all():
            raise ValueError(f"所有标签映射失败！请检查 dataset_name='{dataset_name}' 是否正确，或者 CSV 里的标签是否在 MAP 中。")

        y_part = y_numeric

    # 4. 转换为 Tensor
    try:
        X_np = X_part.values.astype(np.float32)
        y_np = y_part.values.astype(np.float32)
    except ValueError as e:
        print(f"   [致命错误] 无法将数据转换为浮点数: {e}")
        # 打印出有问题的列帮助调试
        print("   检查是否有非数字字符混入特征列...")
        raise e

    # 5. 维度修正
    if X_np.shape[1] != FEATURE_DIM:
        try:
            X_np = X_np.reshape(-1, FEATURE_DIM)
        except ValueError:
            pass

    print(f"   加载完毕: X={X_np.shape}, y={y_np.shape}")
    return torch.from_numpy(X_np), torch.from_numpy(y_np)


