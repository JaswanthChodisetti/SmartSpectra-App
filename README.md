# SmartSpectra: RGB to HSI Reconstruction for Pesticide Screening

## Project Overview
SmartSpectra is a technical framework designed to estimate hyperspectral (HSI) data from standard RGB images of fruit. The goal is to develop a low-cost pre-screening tool that identifies pesticide residues by reconstructing spectral signatures and applying a hierarchical classification pipeline.

### Technical Scope
The system focuses on the 400–1000 nm spectral range. It is important to note that smartphone RGB sensors have hardware-level IR-cut filters that prevent the capture of the NIR (~915-1699 nm) region. Therefore, this project aims to reconstruct the visible and red-edge NIR regions to provide a diagnostic proxy, rather than a full laboratory-grade spectral recovery.

---

## System Architecture

### 1. Pipeline Flow
The processing pipeline is organized as follows:
`RGB Image` $\rightarrow$ `Background Segmentation` $\rightarrow$ `MST++ Reconstruction` $\rightarrow$ `Spectral Preprocessing` $\rightarrow$ `Feature Extraction` $\rightarrow$ `Hierarchical Classification`.

### 2. Core Components
- **Model 1 (Reconstruction)**: Based on the MST++ (Multi-stage Spectral-wise Transformer) architecture. It maps 3-channel RGB inputs to a 31-band hyperspectral cube.
- **Model 2 (Classification)**: A spectral classifier (MLP/XGBoost) trained on a "Pure Science Vector" consisting of SNV, Savitzky-Golay derivatives, and specific spectral ratios.

---

## Repository Structure

```text
Smart_Spectra/
├── demo/               # Streamlit application and inference wrappers
├── scripts/            # Training, evaluation, and data processing scripts
├── models/             # Model checkpoints (.pth and .pkl files)
│   └── checkpoints/    # Final trained weights
├── data/               # Dataset root (Manifests and paired RGB-HSI data)
└── SETUP.md            # Environment and installation guide
```

---

## Methodology & Constraints

### Spectral Preprocessing
To ensure the model is robust against lighting variations and baseline shifts, the following transformations are applied to the reconstructed spectra:
- **Standard Normal Variate (SNV)**: Removes additive and multiplicative scattering effects.
- **Savitzky-Golay 1st Derivative**: Highlights chemical absorption peaks and removes constant baselines.
- **Zonal Analysis**: Extracts features from center, mid, and edge regions of the fruit to account for spatial variance.

### Evaluation Metrics
Reconstruction quality is measured using:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **RMSE** (Root Mean Square Error)
- **MRAE** (Mean Relative Absolute Error)

Classification performance is validated using a **Blind Test Set** to ensure unbiased accuracy and precision reporting.

---

## References
- Cai et al., *MST++: Multi-stage Spectral-wise Transformer for Efficient Spectral Reconstruction*, CVPRW 2022.
- Shi et al., *HSCNN+: Advanced CNN-Based Hyperspectral Recovery from RGB Images*, CVPRW 2018.
- Agro-HSR Dataset: Paired RGB-HSI produce dataset (31 bands, 400-1000 nm).
