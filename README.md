<div align="center">

<h2>From Seeing to Recognising: An Extended Self-Organizing Map for Human Postures Identification (RA-L/ICRA 2025)</h2>

<p>
  <b><a href="https://github.com/qqwwqwq">Xin He</a><sup>1, 2</sup></b> &nbsp;&nbsp;
  <b>Teresa Zielinska<sup>2</sup></b> &nbsp;&nbsp;
  <b>Vibekananda Dutta<sup>2</sup></b> &nbsp;&nbsp;
  <b>Takafumi Matsumaru<sup>1</sup></b> &nbsp;&nbsp;
  <b>Robert Sitnik<sup>2</sup></b>
</p>

<p>
  <sup>1</sup>Waseda University, Japan <br>
  <sup>2</sup>Warsaw University of Technology, Poland
</p>

<p>
  <a href="https://github.com/qqwwqwq?tab=repositories](https://github.com/qqwwqwq/DualSOM"><img src="https://img.shields.io/badge/🏠-Project%20Page-4285F4.svg?style=flat"></a>
  <a href="https://ieeexplore.ieee.org/document/10608412"><img src="https://img.shields.io/badge/📄_IEEE-Paper-b31b1b.svg?style=flat"></a>
  <a href="https://github.com/qqwwqwq/DualSOM/stargazers"><img src="https://img.shields.io/github/stars/qqwwqwq/DualSOM?style=social"></a>
</p>

<hr>
</div>

This repository provides the official implementation of **DualSOM**, a dedicated method for **Human Posture Recognition**, which serves as a foundational step for sequence-based human action recognition. Our framework introduces an **Extended Self-Organized Map (SOM)** combined with a **Sparse Autoencoder (SAE)**. The SAE effectively reduces data dimensionality while strictly preserving essential spatial characteristics. The latent representations are then processed by our extended SOM, which leverages unlabeled data to accurately classify and cluster human postures.

## 🕸️ Network Architecture

<p align="center">
  <img src="./assets/overall_structure.png" width="800">
</p>

## 🚀 Training and Evaluation

You can execute the entire pipeline (Data Loading $\rightarrow$ Sparse Autoencoder $\rightarrow$ DualSOM) using `main.py`. The framework offers a high degree of flexibility through command-line arguments.

### 1. Execution Modes

The repository supports two distinct branches for final evaluation, perfectly matching the concepts proposed in the paper:

**Branch A: Supervised Classification (Default)**
This mode maps the trained Kohonen layer neurons to ground-truth labels and calculates standard classification metrics (Accuracy, Precision, Recall, F1).
```bash
python main.py --run_mode supervised
```

**Branch B: Unsupervised Regrouping (Algorithm 2)**
This mode executes the proposed *Algorithm 2: K-Means for Regrouping the Neurons*. It clusters the trained neurons based on their angular distance without using any labels, and evaluates the performance using external clustering metrics (NMI, AMI, Homogeneity, Completeness).
```bash
# Example: Regrouping the SOM neurons into 5 clusters
python main.py --run_mode unsupervised --n_clusters 5
```

### 2. Training Speed Control (Fast Mode)

By default, the SOM evaluates the validation accuracy every 5 epochs to plot the training curve. If you want to maximize training speed without intermediate evaluations, you can toggle the validation flag:

```bash
# Standard Mode: Evaluates accuracy periodically (Default)
python main.py --som_enable_validation 1

# Fast Mode: Disables intermediate validation for maximum speed
python main.py --som_enable_validation 0
```

### 3. Hyperparameter Configuration

You can easily adjust the network architectures and training parameters via the command line.

**Sparse Autoencoder (SAE) Settings:**
```bash
python main.py \
    --ae_epochs 150 \
    --ae_batch_size 32 \
    --force_train_ae 0  # Set to 1 to force retrain, 0 to load existing weights
```

**Extended SOM Settings:**
```bash
python main.py \
    --som_epochs 50 \
    --som_size_index 10.0 \
    --som_sigma 4.0 \
    --som_lr 0.1
```

### 🎯 Quick Start Example
To run the full unsupervised pipeline at maximum speed with 8 desired clusters:
```bash
python main.py --run_mode unsupervised --n_clusters 8 --som_enable_validation 0
```

## 📧 News
* **[2025.05.21]** 🔥 Our paper is presented at the **2025 International Conference on Robotics and Automation (ICRA)**!
* **[2024.07.12]** 🎉 Our paper is accepted by **IEEE Robotics and Automation Letters (RA-L)**!

## 📜 Reference
If you find our work useful, please consider citing:

```bibtex
@ARTICLE{10608412,
  author={He, Xin and Zielinska, Teresa and Dutta, Vibekananda and Matsumaru, Takafumi and Sitnik, Robert},
  journal={IEEE Robotics and Automation Letters}, 
  title={From Seeing to Recognising–An Extended Self-Organizing Map for Human Postures Identification}, 
  year={2024},
  volume={9},
  number={9},
  pages={7899-7906},
  doi={10.1109/LRA.2024.3433201}
}
