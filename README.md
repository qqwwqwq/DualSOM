# [RA-L/ICRA 2025] DualSOM-Dual-mode-software-for-clustering-and-classification-using-self-organising-map

[**Project Page**](https://github.com/qqwwqwq?tab=repositories) | [**Paper**](https://ieeexplore.ieee.org/document/10608412) 

> **DualSOM-Dual-mode-software-for-clustering-and-classification-using-self-organising-map** <br>
> Xin He, Teresa Zielinska, Vibekananda Dutta, Takafumi Matsumaru, Robert Sitnik
 <br>
> Waseda University and Warsaw University of Technology.



This repository provides the official implementation of our dedicated method for **Human Posture Recognition**, which serves as a foundational step for sequence-based human action recognition. 

Our framework introduces an **Extended Self-Organized Map (SOM)** combined with a **Sparse Autoencoder (SAE)**. The SAE effectively reduces data dimensionality while strictly preserving essential spatial characteristics. The latent representations are then processed by our extended SOM, which leverages unlabeled data to accurately classify and cluster human postures.

### ✨ Key Features & Contributions
* **Extended SOM Architecture:** Integrates an additional layer specifically designed for post-labeling and clustering, providing high resolution in distinguishing complex postures.
* **Task-Oriented Modifications:** Features a custom **angular distance measure** and a specialized **neighborhood function** for weight updates, significantly boosting the SOM's clustering performance.
* **Unsupervised Representation:** Efficiently trains on unlabeled data while maintaining robust discriminative power.
* **Proven Superiority:** Achieves better classification efficiency compared to other representative methods, with comprehensive ablation studies validating the impact of our architectural modifications.

## 🕸️ Network Architecture
<img src="./assets/overall_structure.png" width="800">

## 📧 News
* **[2024.07.12]** 🎉 Our paper is accepted by IEEE ROBOTICS AND AUTOMATION LETTERS！
* **[2025.05.21]** 🔥 Our paper is presented in 2025 International Conference on Robotics and Automation!

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
  keywords={Self-organizing feature maps;Neurons;Training;Decoding;Cameras;Weight measurement;Task analysis;Human postures;improved self-organizing-map;sparse autoencoder;neighbourhood function},
  doi={10.1109/LRA.2024.3433201}}
