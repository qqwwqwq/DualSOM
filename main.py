import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from DaulSOM import DualSom
    from preprocessing import get_dataset
    from sparse_autoencoder import SparseAutoencoder, fit
except ImportError as e:
    print(f"Error: Could not import custom modules ({e}).")
    sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser(description="Human Activity Recognition using Sparse AE + DualSOM")

    # [Added] Dataset selection
    parser.add_argument('--dataset_name', type=str, default='wut',
                        choices=['wut', 'pku'],
                        help="Select the dataset to use: 'wut' for Warsaw dataset | 'pku' for PKU dataset")

    # Path parameters
    parser.add_argument('--train_path', type=str,
                        default="/Datas/WUT/WUT_data_train/")
    parser.add_argument('--test_path', type=str,
                        default="/Datas/WUT/WUT_data_test/")
    parser.add_argument('--output_dir', type=str, default="./results")
    parser.add_argument('--reduction_factor', type=int, default=1)

    # AE parameters
    parser.add_argument('--ae_batch_size', type=int, default=32)
    parser.add_argument('--ae_epochs', type=int, default=150)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--force_train_ae', type=int, default=0,
                        help="0: Load model if exists (default) | 1: Force train Autoencoder")

    # SOM parameters
    parser.add_argument('--som_size_index', type=float, default=10)
    parser.add_argument('--som_epochs', type=int, default=50)
    parser.add_argument('--som_sigma', type=float, default=4)
    parser.add_argument('--som_lr', type=float, default=0.1)

    # [Modified] Default set to 1: Enable validation calculation during training to plot the accuracy curve
    parser.add_argument('--som_enable_validation', type=int, default=1,
                        help="1: Enable validation during training (default) | 0: Fast mode")

    # ==========================================
    # [Modified] Default mode changed to 'supervised' (classification validation with labels)
    # ==========================================
    parser.add_argument('--run_mode', type=str, default='supervised',
                        choices=['supervised', 'unsupervised'],
                        help="Run mode: 'supervised' (default, classification with labels) | 'unsupervised' (Algorithm 2 unsupervised clustering from the paper)")
    parser.add_argument('--n_clusters', type=int, default=5,
                        help="Number of desired clusters K_m for Algorithm 2")

    return parser.parse_args()


# =====================================================================
# [New Module] Strictly replicate Algorithm 2 from the paper: K-Means for Regrouping the Neurons
# =====================================================================
class SOMClusterer:
    def __init__(self, n_clusters, max_iter=100, threshold=1e-4):
        self.K = n_clusters
        self.max_iter = max_iter
        self.threshold = threshold
        self.centroids = None
        self.labels_map = None

    def _angular_distance_matrix(self, W1, W2):
        """Calculates pairwise angular distance exactly as in DualSom"""
        W1_norm = W1 / (np.linalg.norm(W1, axis=1, keepdims=True) + 1e-8)
        W2_norm = W2 / (np.linalg.norm(W2, axis=1, keepdims=True) + 1e-8)
        sim = np.clip(np.dot(W1_norm, W2_norm.T), -1.0, 1.0)
        ang = np.arccos(sim) * 180 / np.pi
        ang = np.minimum(ang, 180 - ang)
        return ang / 180 * np.pi

    def fit(self, som_weights):
        grid_x, grid_y, features = som_weights.shape
        U = grid_x * grid_y

        # 1. Vectorised weights vw_u from Kohonen layer
        vw = som_weights.reshape(U, features)

        # 2. primary array of distances dis[i, j]
        dis = self._angular_distance_matrix(vw, vw)
        np.fill_diagonal(dis, 0)

        # 3. Obtain the first centroid c1
        c1 = np.argmin(np.sum(dis, axis=0))
        centroids_idx = [c1]

        # initialise delta^1
        delta = dis[c1, :].copy()

        # Obtain other centroids
        for i in range(1, self.K):
            last_c = centroids_idx[-1]
            # Produce new delta
            delta = np.minimum(delta, dis[last_c, :])
            next_c = np.argmax(delta)
            centroids_idx.append(next_c)

        self.centroids = vw[centroids_idx].copy()

        # 4. Obtain final clusters (Iterative K-Means)
        labels = np.zeros(U, dtype=int)
        for it in range(self.max_iter):
            # Assign remaining neurons to their nearest prototype clusters
            dist_to_centroids = self._angular_distance_matrix(vw, self.centroids)
            labels = np.argmin(dist_to_centroids, axis=1)

            # Calculate coordinates of cluster centroid prototypes
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.K):
                cluster_points = vw[labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[k] = self.centroids[k]

            # Check threshold difference
            diff = np.max(np.linalg.norm(new_centroids - self.centroids, axis=1))
            self.centroids = new_centroids
            if diff < self.threshold:
                print(f"      [Algorithm 2] Converged at iteration {it + 1}")
                break

        # Map 1D labels back to 2D SOM grid
        self.labels_map = labels.reshape(grid_x, grid_y)
        return self

    def predict(self, som, data_X):
        preds = []
        for x in data_X:
            win_x, win_y = som.winner(x)
            cluster_id = self.labels_map[win_x, win_y]
            preds.append(cluster_id)
        return np.array(preds)


def classify_som(som, data, winmap):
    from collections import Counter
    if not winmap:
        return [0] * len(data)
    default_class = sum(winmap.values(), Counter()).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result


def load_and_process_data(args):
    print(">>> Loading Data...")

    # 1. Dynamically get dataset name (fallback to 'wut' if not in args)
    dataset_name = args.dataset_name.lower()

    # 2. Allocate different cache directories based on dataset name
    if dataset_name == 'pku':
        data_dir = "Datas/PKU/Preprocessed_data"
    else:
        data_dir = "Datas/WUT/Preprocessed_data"

    # Ensure cache directory exists to prevent errors when saving csv later
    os.makedirs(data_dir, exist_ok=True)

    train_processed_path = os.path.join(data_dir, "train_data.csv")
    test_processed_path = os.path.join(data_dir, "test_data.csv")

    trainData, trainlabel = get_dataset(raw_path=args.train_path,
                                        processed_path=train_processed_path,
                                        dataset_name=dataset_name)
    testData, testlabel = get_dataset(raw_path=args.test_path,
                                      processed_path=test_processed_path,
                                      dataset_name=dataset_name)

    red = args.reduction_factor
    n_train = int(trainData.shape[0] / red)
    n_test = int(testData.shape[0] / red)

    X_train_raw = trainData[:n_train].clone().detach().cpu().numpy()
    X_test_raw = testData[:n_test].clone().detach().cpu().numpy()
    y_train = trainlabel.cpu().numpy()[:n_train]
    y_test = testlabel.cpu().numpy()[:n_test]

    print(f"Normalizing INPUT data...")
    input_scaler = MinMaxScaler()
    X_train_scaled = input_scaler.fit_transform(X_train_raw)
    X_test_scaled = input_scaler.transform(X_test_raw)

    X_train = torch.from_numpy(X_train_scaled).float()
    X_test = torch.from_numpy(X_test_scaled).float()

    print(f"Data Loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Dynamically save Ground Truth filename to prevent overwriting between datasets
    np.save(os.path.join(args.output_dir, f"{dataset_name}_gt.npy"), y_test)

    return X_train, y_train, X_test, y_test


def run_autoencoder(X_train, X_test, args):
    print(f">>> Processing Sparse Autoencoder on {args.device}...")
    save_path = os.path.join('weight', f'pretrain_AE_batch_{args.ae_batch_size}_2026_71.69.pth')
    model = SparseAutoencoder().to(args.device)

    should_train = (args.force_train_ae == 1) or not os.path.exists(save_path)

    if should_train:
        print(f"Starting training (force_train_ae={args.force_train_ae})...")
        model = fit(model, args.ae_batch_size, X_train, args.ae_epochs, args.device, save_path)
    else:
        print(f"Loading pre-trained model from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=args.device))

    # Process in batches during feature extraction to prevent OOM
    def extract_features_batched(data_tensor):
        model.eval()
        dataset = TensorDataset(data_tensor)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        features_list = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(args.device)
                feat = model.fd(x)
                features_list.append(feat.cpu().numpy())
        return np.vstack(features_list)

    print("Extracting features...")
    X_train_ae = extract_features_batched(X_train)
    X_test_ae = extract_features_batched(X_test)

    print("Normalizing FEATURE data for SOM...")
    feature_scaler = StandardScaler()
    X_train_ae = feature_scaler.fit_transform(X_train_ae)
    X_test_ae = feature_scaler.transform(X_test_ae)

    return X_train_ae, X_test_ae, model


def run_som_training_and_eval(X_train, y_train, X_test, y_test, args):
    N = X_train.shape[0]
    M = X_train.shape[1]

    size = math.ceil(np.sqrt(args.som_size_index * np.sqrt(N)))
    print(f">>> SOM Config: Grid Size={size}x{size}, Max Iter={args.som_epochs * N}, Sigma={args.som_sigma}")

    som = DualSom(size, size, M, sigma=args.som_sigma, learning_rate=args.som_lr,
                  repulsion=2, neighborhood_function='bubble',
                  activation_distance='angular')

    som.pca_weights_init(X_train)

    max_iter = args.som_epochs * N
    if args.som_enable_validation == 1:
        print("    -> Standard Mode (with periodic validation)...")
        som.train_batch(X_train, y_train, X_test, y_test, max_iter, verbose=True, enable_validation=True)
    else:
        print("    -> Fast Mode (Skipping validation)...")
        som.train_batch(X_train, max_iter, verbose=True, enable_validation=False)

    print("\n>>> SOM Training Finished! Executing Evaluation Branch...")

    # =======================================================
    # [Core Addition] Dynamically dispatch tasks based on run_mode
    # =======================================================
    if args.run_mode == 'supervised':
        print("\n=== Branch: Supervised Classification ===")
        winmap = som.labels_map(X_train, y_train)
        y_pred = classify_som(som, X_test, winmap)

        print("\nClassification Report:")
        print(classification_report(y_test, np.array(y_pred)))

        # Calculate all supervised classification metrics
        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average='macro', zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
            "nmi": metrics.normalized_mutual_info_score(y_test, y_pred),
            "ami": metrics.adjusted_mutual_info_score(y_test, y_pred)
        }

    elif args.run_mode == 'unsupervised':
        print(f"\n=== Branch: Unsupervised Algorithm 2 Regrouping (K_m={args.n_clusters}) ===")
        clusterer = SOMClusterer(n_clusters=args.n_clusters)

        print("    -> Fitting Algorithm 2 onto SOM weights...")
        clusterer.fit(som._weights)

        print("    -> Predicting test set clusters...")
        y_pred = clusterer.predict(som, X_test)

        # In unsupervised mode, true class IDs and cluster IDs may not align, so Accuracy is not calculated
        # Instead, we calculate external clustering metrics invariant to permutation (NMI, AMI, Homogeneity, Completeness)
        metrics_dict = {
            "nmi": metrics.normalized_mutual_info_score(y_test, y_pred, average_method='arithmetic'),
            "ami": metrics.adjusted_mutual_info_score(y_test, y_pred, average_method='arithmetic'),
            "homogeneity": metrics.homogeneity_score(y_test, y_pred),
            "completeness": metrics.completeness_score(y_test, y_pred)
        }

    # Print the corresponding metrics
    print("\n" + "=" * 30 + "\nMetrics Summary\n" + "=" * 30)
    for k, v in metrics_dict.items():
        print(f"{k.capitalize()}: {v:.4f}")

    # [Modified] Dynamically save prediction filename with dataset name suffix
    np.save(os.path.join(args.output_dir, f"{args.dataset_name}_pred_{args.run_mode}.npy"), y_pred)
    visualize_results(X_test, y_test, y_pred, som, args)

    return som, metrics_dict


def visualize_results(X_test, y_test, y_pred, som, args):
    print(">>> Generating Visualizations...")
    limit = 1000
    if len(X_test) > limit:
        idx = np.random.choice(len(X_test), limit, replace=False)
        H = X_test[idx]
        C_pred = np.array(y_pred)[idx]
        gd = np.array(y_test)[idx]
    else:
        H = X_test
        C_pred = np.array(y_pred)
        gd = np.array(y_test)

    try:
        H = np.array(H, dtype=np.float32)
        gd = np.array(gd).astype(float).astype(int)
        C_pred = np.array(C_pred).astype(float).astype(int)
    except Exception as e:
        print(f"!!! Data conversion error: {e}")
        return

    print("Running t-SNE...")
    try:
        tsne = TSNE(n_components=2, init='random', learning_rate=200.0, random_state=42)
        H_2D = tsne.fit_transform(H)
    except Exception as e:
        print(f"t-SNE failed: {e}")
        return

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    sc1 = plt.scatter(H_2D[:, 0], H_2D[:, 1], s=10, c=C_pred, cmap='tab10', alpha=0.7)
    plt.colorbar(sc1, label='Predicted Class / Cluster')
    plt.title(f"Prediction ({args.run_mode.capitalize()})")

    plt.subplot(1, 3, 2)
    sc2 = plt.scatter(H_2D[:, 0], H_2D[:, 1], s=10, c=gd, cmap='tab10', alpha=0.7)
    plt.colorbar(sc2, label='True Label')
    plt.title(f"Ground Truth ({args.dataset_name.upper()})")

    plt.subplot(1, 3, 3)
    if hasattr(som, 'acuracy') and len(som.acuracy) > 0:
        plt.plot(som.it, som.acuracy)
        plt.title("SOM Training Accuracy")
        step = max(1, len(som.it) // 10)
        for a, b in zip(som.it[::step], som.acuracy[::step]):
            plt.text(a, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
    else:
        plt.text(0.5, 0.5, "Fast Mode\n(No Accuracy Curve)", ha='center', va='center')

    os.makedirs(args.output_dir, exist_ok=True)
    # [Modified] Dynamically save visualization image filename
    save_path = os.path.join(args.output_dir, f'visualization_{args.dataset_name}_{args.run_mode}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")


if __name__ == "__main__":
    args = get_args()
    X_train_raw, y_train, X_test_raw, y_test = load_and_process_data(args)
    X_train_ae, X_test_ae, ae_model = run_autoencoder(X_train_raw, X_test_raw, args)
    run_som_training_and_eval(X_train_ae, y_train, X_test_ae, y_test, args)
    print(">>> All Done.")
