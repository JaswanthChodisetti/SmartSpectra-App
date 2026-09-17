# Code Repository

This directory contains the source code for the SmartSpectra project.

## Setup Instructions

To get started with the RGB→HSI reconstruction (MST++ model), follow these steps:

### 1. Clone the MST++ Repository
```bash
git clone https://github.com/caiyuanhao1998/MST-plus-plus.git
cd MST-plus-plus
```

### 2. Install Dependencies
```bash
pip install -q opencv-python einops scipy h5py hdf5storage tqdm gdown
```

### 3. Download Pretrained Weights
```bash
mkdir -p predict_code/model_zoo
gdown --id 18X6RkcQaIuiV5gRbswo7GLv7WJG9M_WM -O predict_code/model_zoo/mst_plus_plus.pth
```

### 4. Test with Sample Image
```bash
cd predict_code
python test.py --method mst_plus_plus \
  --pretrained_model_path ./model_zoo/mst_plus_plus.pth \
  --rgb_path ./demo/ARAD_1K_0912.jpg \
  --outf ./exp/mst_plus_plus/ \
  --gpu_id 0
```

### 5. Visualize Results
See the Week1_Setup.ipynb notebook for detailed visualization code.

## Project-Specific Code

As you progress through the project phases, you'll add:

### Phase 2: Dataset Filtering Scripts
- Scripts to help identify fruit-relevant scenes from the ARAD-1K dataset
- Tools to create custom train/validation splits

### Phase 3: Fine-tuning Scripts
- Modified training scripts for fine-tuning MST++ on fruit-specific data
- Configuration files for adjusted hyperparameters (lower LR, smaller batch size, fewer epochs)

### Phase 4: Evaluation Scripts
- Metrics computation (MRAE, RMSE)
- Comparison scripts (pretrained vs. fine-tuned performance)
- Visualization tools for spectral signature comparison

### Model 2: Quality Prediction
- HSI-to-quality prediction models
- End-to-end pipeline integration (RGB → reconstructed HSI → quality prediction)

## Directory Structure (within code/)
```
code/
├── reconstruction/         # RGB→HSI reconstruction code
│   ├── MST-plus-plus/      # Cloned MST++ repository
│   └── scripts/            # Custom scripts for dataset prep, training, eval
└── quality_prediction/     # HSI→quality prediction models
    ├── models/             # Quality prediction model implementations
    └── scripts/            # Training and evaluation scripts
```