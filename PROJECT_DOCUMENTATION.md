# SmartSpectra: Precision Hyperspectral AI Triage for Pesticide Detection

## 📖 Project Overview
SmartSpectra is a scientific instrument designed to detect chemical pesticide residues on produce (specifically apples) using a two-stage AI pipeline. The system transforms a standard RGB image into a reconstructed hyperspectral cube to identify chemical signatures that are invisible to the human eye and standard cameras.

### The Core Pipeline
**RGB Image** $\rightarrow$ **Model 1 (MST++)** $\rightarrow$ **Reconstructed HSI** $\rightarrow$ **Feature Extraction** $\rightarrow$ **Model 2 (XGBoost)** $\rightarrow$ **Prediction**

---

## 🏗️ System Architecture

### 1. Model 1: RGB $\rightarrow$ HSI Reconstruction
Model 1 uses the **MST++ (Model-based Spectral Transformer Plus Plus)** architecture to recover spectral information from a single RGB image.

- **Purpose**: Reconstruct a 31-band hyperspectral cube (covering 400nm to 1000nm).
- **Architecture**: `MST_Plus_Plus` (found in `code/MST-plus-plus/train_code/architecture/MST_Plus_Plus.py`).
- **Input**: $256 \times 256 \times 3$ RGB image, min-max normalized.
- **Output**: $256 \times 256 \times 31$ reconstructed hyperspectral cube.
- **Training Dataset**: **AGRO-HSR** (Agricultural Hyperspectral Dataset).
- **Training Configuration**:
  - **Loss Function**: `Loss_L1` (Primary) + `Loss_Smoothness` (Spatial TV) + `Loss_SpectralSmoothness` (Spectral TV).
  - **Regularization**: $\lambda_{smooth}=0.02$, $\lambda_{spectral}=0.02$.
  - **Optimizer**: Adam with `CosineAnnealingLR`.
  - **Initial LR**: $4 \times 10^{-4}$.
  - **Batch Size**: 20.
  - **Best Checkpoint**: `models/checkpoints/run_trained/net_best.pth`.
- **Evaluation Metrics**: Mean Relative Absolute Error (MRAE), RMSE, and PSNR.

### 2. Model 2: Detection & Classification
Model 2 is a highly regularized classifier that operates on a "Pure Science" feature vector extracted from the reconstructed HSI cube.

- **Purpose**: Classify produce as **Fresh**, **Fungicide**, or **Insecticide**.
- **Architecture**: XGBoost Classifier.
- **The Hybrid Feature Vector (104 Dimensions)**:
  - **SNV (Standard Normal Variate)** [31 dims]: Corrects for scattering and baseline curvature.
  - **Savitzky-Golay 1st Derivative** [30 dims]: Isolates chemical absorbance slopes.
  - **Spectral Standard Deviation (STD)** [31 dims]: Captures spatial variability across the fruit.
  - **Targeted Spectral Ratios** [12 dims]: Indices tuned to the Deep NIR (800-1000nm) zone.
- **Training Strategy (The "Sweetspot")**:
  - **Strict Regularization**: `max_depth=2` to prevent overfitting to image textures.
  - **Class Weighting**: $1.75\times$ multiplier for the "Fresh" class to penalize False Positives.
- **Final Checkpoint**: `models/checkpoints/model2_final.pkl`.

---

## 📊 Datasets & Data Management

### Model 1 Data
- **AGRO-HSR**: Used for pre-training MST++ on agricultural hyperspectral data.

### Model 2 Data
- **Internal Dataset**: Apple samples categorized into Fresh, Low Concentration, and High Concentration of pesticides.
- **Blind Test Dataset (Vaishnavi Dataset)**: A completely unseen external dataset used to verify real-world generalizability.
- **The Vault**: A secure directory (`data/The_Vault`) containing images never used in training or validation to ensure zero data leakage.

### Dataset Verification
- **Labels**: Mapped based on filename prefixes (e.g., `ma-` for Insecticide, `a-` for Fungicide).
- **Splits**: Training, Validation, and a strict Blind Test set.

---

## 🔬 Spectral Range & Scientific Validity

### The NIR Challenge
The system targets the **Deep NIR (800-1000nm)** region, where pesticide residues leave distinct absorbance fingerprints.

- **RGB Limitation**: Standard RGB cameras only capture visible light. Reconstructing wavelengths beyond the camera's spectral response is an "ill-posed" problem.
- **Scientific Defense**: Model 1 learns a mapping from RGB textures and colors to spectral signatures based on the AGRO-HSR dataset. While not a replacement for a true HSI sensor, the reconstructed features are sufficient for classification when combined with the "Sweetspot" regularization of Model 2.
- **Generalization**: The 81% accuracy on the external Vaishnavi dataset proves that the system has learned a generalizable spectral proxy rather than just memorizing the training set.

---

## 📈 Experiments & Results

### Model Performance Summary
| Metric | Internal Validation | External Blind Test |
| :--- | :--- | :--- |
| **Accuracy** | **85.3%** | **81.0%** |
| **Pesticide Precision** | 95.0% | ~95% |
| **Generalization Gap** | - | $\approx 4.3\%$ |

### Key Findings
- **Explainability**: Using SHAP (SHapley Additive exPlanations), we verified that the model's decisions are driven by specific bands in the 800-1000nm range.
- **Stability**: The "Sweetspot" model (`max_depth=2`) eliminated "Pesticide Collapse" (where the model predicts everything as Pesticide).

---

## 🖥️ Streamlit Demo

### Functionality
- **Diagnostic Mode**: Upload an RGB image $\rightarrow$ MST++ Reconstruction $\rightarrow$ Triage Result $\rightarrow$ Confidence Score.
- **Science Deep-Dive**: View the reconstructed spectrum vs. a Fresh Apple baseline.
- **Analysis Suite**: Review global validation metrics and SHAP importance plots.

### Execution
```bash
# 1. Activate environment
source venv/bin/activate

# 2. Run the app
streamlit run demo/app.py
```

---

## 📂 Repository Structure
```text
Smart_Spectra/
├── code/
│   └── MST-plus-plus/       # Model 1 architecture and training code
├── data/
│   ├── clean_dataset/       # Curated training samples
│   ├── pesticides/          # Raw pesticide datasets
│   └── The_Vault/           # Strictly unseen images for final validation
├── demo/
│   ├── app.py              # Streamlit UI
│   ├── inference.py        # Full pipeline logic (Model 1 + Model 2)
│   └── routing.py          # Image triage/routing logic
├── models/
│   └── checkpoints/        # .pth (Model 1) and .pkl (Model 2) weights
├── scripts/
│   ├── blind_test_generic.py # External dataset evaluation
│   └── generate_faculty_visuals.py # SHAP and Confusion Matrix plots
└── Faculty_Report.md        # Final scientific documentation
```

---

## ⚠️ Limitations & Known Issues
- **Reconstruction Fidelity**: Reconstructed HSI is a proxy; it cannot replace a physical HSI sensor for quantitative chemical analysis.
- **Spectral Range**: The current model is limited to 1000nm. Detection of chemicals requiring $>1000$ nm (e.g., certain organic bonds at 1500nm) is not possible.
- **Dataset Size**: Model 2 is trained on a relatively small set of apple samples; expanding to other fruits (oranges, grapes) requires new data.
- **Domain Shift**: Performance may vary based on lighting conditions and apple varieties.

---

## 🛠️ Reproducibility

### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Pipeline Execution
1. **Model 1 Inference**: `python demo/inference.py --rgb <image.jpg>`
2. **Full Triage**: Run `streamlit run demo/app.py` and upload an image.

---

## 🏁 Project Status

| Component | Status | Evidence | Notes |
| :--- | :--- | :--- | :--- |
| **Model 1** | ✅ Completed | `net_best.pth` | MST++ Reconstruction stable. |
| **Model 2** | ✅ Completed | `model2_final.pkl` | Sweetspot XGBoost reached 81% blind acc. |
| **Pipeline** | ✅ Completed | `inference.py` | End-to-end RGB $\rightarrow$ Prediction. |
| **Demo UI** | ✅ Completed | `app.py` | Professional Glassmorphism UI. |
| **Evaluation** | ✅ Completed | `benchmark_data.json` | Verified on external Vaishnavi dataset. |

### Final Achievements
- **Completed**: Full pipeline implementation, "Sweetspot" regularization, SHAP explainability, and professional UI.
- **Experimented With**: Fine-tuning on SpectroFood, synthetic data generation.
- **Planned**: Expansion to other produce types, integration with physical HSI sensors.
