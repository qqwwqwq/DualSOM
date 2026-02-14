<div align="center">

<h2>From Seeing to Recognising: An Extended Self-Organizing Map for Human Postures Identification (RA-L/ICRA 2025)</h2>

<p>
  <b><a href="https://github.com/qqwwqwq">Xin He</a><sup>1, 2</sup></b> \t
  <b>Teresa Zielinska<sup>2</sup></b> \t
  <b>Vibekananda Dutta<sup>2</sup></b> \t
  <b>Takafumi Matsumaru<sup>1</sup></b> \t
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

## 📧 News
* **[2025.05.21]** 🔥 Our paper will be presented at the **2025 International Conference on Robotics and Automation (ICRA)**!
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
