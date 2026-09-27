# Setup Guide: SmartSpectra Project

This guide ensures that the SmartSpectra environment is configured correctly to run the RGB $\rightarrow$ HSI reconstruction and classification pipeline without errors.

## 🛠️ 1. Environment Requirements
- **Python Version**: 3.10 or higher (recommended)
- **Hardware**: NVIDIA GPU with CUDA support is recommended for faster inference, but the code will automatically fallback to CPU if no GPU is found.

## 📦 2. Installing Dependencies
The easiest way to install all required libraries is to use the provided `requirements.txt` file.

Run the following command in your terminal:
```bash
pip install -r requirements.txt
```

## 📂 3. Folder Structure & Data
Ensure your project folder is structured as follows:
```text
Smart_Spectra/
├── code/
│   └── MST-plus-plus/       # The reconstruction architecture
├── data/                    # Your HSI cubes and RGB images
├── models/
│   └── checkpoints/         # .pth and .pkl model weights
├── demo/                    # The Streamlit app
└── scripts/                 # Evaluation and training scripts
```

## 🚀 4. Running the Project

### Running the Demo App
To launch the interactive Streamlit interface:
```bash
streamlit run demo/app.py
```

### Running the Blind Test Evaluation
To verify the model performance on the blind test manifest:
```bash
python scripts/blind_test_manifest.py data/pesticides/vaishnavi_2023/apple_pairs_extended_blind.json
```

## ⚠️ Critical Troubleshooting (Common Errors)

### "ModuleNotFoundError: No module named 'MST_Plus_Plus'"
The model architecture is located in a nested folder. The scripts are designed to handle this automatically by adding the path to `sys.path`. However, if you encounter this error, ensure the `code/MST-plus-plus/train_code/architecture` folder exists.

### "FileNotFoundError" for Weights
If the app cannot find the models, verify that your weights are placed in:
`models/checkpoints/run_trained/net_best.pth`
and
`models/checkpoints/model2_final.pkl`

### "hdf5storage" Errors
If you encounter errors loading `.mat` files, ensure `hdf5storage` is installed:
```bash
pip install hdf5storage
```
