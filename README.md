# Music Source Separation Using an AutoEncoder

Sapienza University of Rome - Advanced Machine Learning - 2024/25/1 - Final Project

This project explores **Music Source Separation (MSS)** by employing a **state-of-the-art autoencoder** to separate instrument tracks from mixed audio files. Our approach focuses on leveraging clustering algorithms to extract distinct instrument components from the bottleneck layer of the autoencoder, offering a novel perspective in music source separation research.

---

## Table of Contents
- [Introduction](#introduction)
- [Dataset and Pretrained Models](#dataset-and-pretrained-models)
- [Methodology](#methodology)
  - [AutoEncoder Architecture](#autoencoder-architecture)
  - [Clustering Algorithms](#clustering-algorithms)
- [Metrics for Evaluation](#metrics-for-evaluation)
- [Results](#results)
- [Future Directions](#future-directions)
- [Contributors](#contributors)
- [References](#references)

---

## Introduction

Music Source Separation (MSS) aims to isolate individual instrument tracks from a mixed audio file. Current methods often rely on ground truth data for supervised learning, which limits their generalizability. This project utilizes an **AutoEncoder architecture** to process and encode mixed audio, followed by **unsupervised clustering** in the latent space to separate instruments, even when the exact number of instruments is unknown.

Our primary focus is on the **others** category of instrument stems, as provided by the MUSDB18 dataset, and achieving further separation into individual components like piano, guitar, and synthesizers.

---

## Dataset and Pretrained Models

- **Dataset**: We utilized the **MUSDB18 dataset**, which contains 150 songs split into stems for drums, bass, vocals, and others.
  - **Training Set**: 130 songs from the "others" category.
  - **Test Set**: 20 songs from the "others" category.

- **Pretrained Model**: The initial separation was performed using **Open-Unmix**, a pretrained source separation model.

---

## Methodology

### AutoEncoder Architecture
We employed the **SEANet EnCodec AutoEncoder** to encode audio into a compressed latent space and reconstruct it. The training objective combines:
- **Mean Squared Error (MSE)** in the time and frequency domains.
- **KL Divergence**: To enforce a structured Gaussian latent space that clusters similar features (e.g., instruments) together.

Key findings:
- Adding the KL Divergence term improved the latent space structure, facilitating better separation in downstream clustering.

### Clustering Algorithms
Clustering was applied to the latent embeddings to assign frames to clusters representing different instruments. We experimented with:
1. **K-Means**
2. **Agglomerative Clustering**
3. **DBSCAN**

Clusters were decoded to reconstruct separated audio sources.

---

## Metrics for Evaluation

The quality of separation was evaluated using several metrics:

1. **Reconstruction Error (MSE)**: Measures the difference between the original mixture and the sum of reconstructed clusters.
2. **Cluster Entropy**: Indicates how evenly frames are distributed across clusters. Lower entropy suggests better separation.
3. **Sparsity and Energy Distribution**: Evaluates whether energy is concentrated within each cluster, indicating distinct sources.
4. **Spectrogram and Visualization**: Qualitative analysis of cluster purity via time–frequency visualizations.
5. **Signal-to-Distortion Ratio (SDR)**: Quantifies the accuracy of separation relative to target signals (in synthetic datasets).

---

## Results

- **AutoEncoder Training**:
  - Final loss after 24 epochs: **0.1822**
  - Metrics on test set:
    - Mean Squared Error (MSE): **0.0024**
    - Signal-to-Noise Ratio (SNR): **1.7897 dB** (vs. **5.32 dB** for Open-Unmix)
    - Spectral MSE: **13.6975**
    - Cosine Similarity: **0.7734**

- **Clustering Comparison**:
  - **Agglomerative Clustering**: Best overall performance with balanced energy distribution across clusters.
  - **K-Means**: Produced comparable results but struggled in some configurations.
  - **DBSCAN**: Demonstrated good sparsity but formed fewer clusters in some cases.

| Method         | MSE     | Cluster 1 Entropy | Cluster 1 Energy (%) | Cluster 2 Energy (%) |
|----------------|---------|--------------------|-----------------------|-----------------------|
| Agglomerative  | 0.0164  | 6.4309             | 52.39                | 13.15                |
| K-Means        | 0.0184  | 6.5551             | 18.88                | 46.29                |
| DBSCAN         | 0.00775 | 6.3847             | 67.42                | N/A                  |

---

## Future Directions

1. **Improved Loss Functions**:
   - Experimenting with perceptual loss to better capture high-level audio features.
   - Adding decay terms to regularize the latent space further.

2. **Extended Training**:
   - Training the autoencoder for more epochs to explore further loss reduction and latent space refinement.

3. **Enhanced Clustering**:
   - Investigating more advanced clustering techniques, such as contrastive loss and deep clustering methods, to improve separation performance.

4. **Real-World Data**:
   - Applying the methodology to more diverse datasets for better generalizability.

---

## Group Members
- **Anja Stanić**: 2190471
- **Francesco Brigante**: 1987197
- **Giorgia Barboni**: 1885285
- **Murad Hüseynov**: 2181584

---

## References

1. [Open-Unmix: A Reference Implementation for Music Source Separation](https://github.com/sigsep/open-unmix-pytorch)  
2. [SEANet EnCodec: High-Fidelity Neural Audio Compression](https://github.com/facebookresearch/encodec)  
3. Julian Neri et al., "Unsupervised Blind Source Separation with Variational Auto-Encoders," 2021.  
4. Lin et al., "Unsupervised Harmonic Sound Source Separation with Spectral Clustering," 2020.  
5. Hershey et al., "Deep Clustering: Discriminative Embeddings for Segmentation and Separation," 2016.  

