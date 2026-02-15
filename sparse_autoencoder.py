import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.style.use('ggplot')


# --- Model Definition ---
class SparseAutoencoder(nn.Module):
    def __init__(self):
        super(SparseAutoencoder, self).__init__()
        self.enc1 = nn.Linear(57, 72)
        self.bn1 = nn.BatchNorm1d(72)  # Add BN to accelerate convergence

        self.enc2 = nn.Linear(72, 36)
        # No BN in the hidden layer to allow the L1 penalty to take effect

        self.dec1 = nn.Linear(36, 72)
        self.bn2 = nn.BatchNorm1d(72)

        self.dec2 = nn.Linear(72, 57)

    def forward(self, x):
        # Encoding phase
        x = F.celu(self.bn1(self.enc1(x)))
        latent = F.relu(self.enc2(x))  # [Key Improvement] ReLU produces absolute zeros, perfectly matching the L1 sparsity penalty

        # Decoding phase
        x = F.celu(self.bn2(self.dec1(latent)))
        reconstructed = torch.sigmoid(self.dec2(x))  # [Key Improvement] Sigmoid perfectly matches the [0,1] input from MinMaxScaler

        return reconstructed, latent

    def fd(self, x):
        x = F.celu(self.bn1(self.enc1(x)))
        x = F.relu(self.enc2(x))
        return x


# --- Auxiliary Classes and Functions ---
class History:
    def __init__(self, target='min'):
        self.value = None
        self.best = float('inf') if target == 'min' else 0.
        self.n_no_better = 0
        self.better = False
        self.target = target
        self.history = []

    def add(self, value):
        if self.target == 'min' and value < self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        elif self.target == 'max' and value > self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        else:
            self.n_no_better += 1
            self.better = False
        self.value = value
        self.history.append(value.item() if torch.is_tensor(value) else value)


def sparse_loss(latent_code):
    """
    [Fix] Corrected a critical logical error:
    Directly receives the latent vector generated during forward propagation,
    and applies the sparsity penalty (L1) exclusively to it.
    This ensures the consistency of the activation function and avoids any redundant computational waste.
    """
    loss = torch.mean(torch.abs(latent_code))
    return loss


# --- Core Functionality: Training Function ---
def fit(model, batch_size, tr_data, epochs, device, save_path):
    print(f'Starting training: Batch Size={batch_size}, Epochs={epochs}')
    add_sparsity = 'yes'
    reg_param = 0.001

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    criterion = nn.MSELoss()

    # [Optimization] In conjunction with the learning rate scheduler, the initial lr can be slightly increased to 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # [Optimization] Dynamic learning rate decay: halve the learning rate when Loss stops decreasing for 10 consecutive epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    tr_loss_h = History('min')

    n_sample = tr_data.shape[0]
    n_batch = (n_sample + batch_size - 1) // batch_size

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        # Shuffle data
        perm = torch.randperm(n_sample)
        tr_data = tr_data[perm]

        for i in range(n_batch):
            # Handle boundary conditions when fetching batches
            end_idx = min((i + 1) * batch_size, n_sample)
            # Skip if the batch has only 1 sample, as BatchNorm1d will throw an error
            if end_idx - i * batch_size <= 1:
                continue

            x = tr_data[i * batch_size: end_idx].to(device)
            optimizer.zero_grad()

            outputs, latent = model(x)

            # Calculate reconstruction error
            mse_loss = criterion(outputs, x)

            # Calculate sparsity error
            if add_sparsity == 'yes':
                l1_loss = sparse_loss(latent)
                loss = mse_loss + reg_param * l1_loss
            else:
                loss = mse_loss

            loss.backward()

            # [Optimization] Gradient clipping: prevents gradient explosion in early training stages (sudden Loss spikes)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        # Calculate the average Loss for the entire Epoch
        epoch_loss = running_loss / n_batch
        tr_loss_h.add(epoch_loss)

        # [Optimization] Learning rate scheduler steps based on epoch_loss
        scheduler.step(epoch_loss)

        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss * 10000:.4f} (Best: {tr_loss_h.best * 10000:.4f}) | LR: {current_lr:.6f}")

    print('End AutoEncoder training!')

    # Save the final model after training
    torch.save(model.state_dict(), save_path)
    print(f"--> Final model saved to {save_path}")

    return model


# --- New Feature: Load or Train ---
def get_or_train_model(tr_data, epochs=50, batch_size=64, force_train=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseAutoencoder()
    save_path = f'weight/sparse_ae_batch_{batch_size}.pth'

    if os.path.exists(save_path) and not force_train:
        print(f"Found saved model at {save_path}. Loading...")
        # Load weights
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.to(device)
    else:
        if force_train:
            print("Force train is True. Starting training...")
        else:
            print("No saved model found. Starting training...")
        model = fit(model, batch_size, tr_data, epochs, device, save_path)

    return model





