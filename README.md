# AI Driven Calibration of Low Cost Air Quality Sensors Using MLP and LSTM

This repository contains a Master's level project for calibrating drifted PM2.5 air quality sensors using Deep Learning. The project follows a **Spiral Development Approach**, evolving from baseline neural networks to advanced LSTM architectures.

## 📁 Repository Contents
* `project_code.py`: Full Python implementation including Iteration 1 (Baseline) and Iteration 2 (Advanced).
* `.gitignore`: Configured to manage large files and local environments.

## 📊 Dataset Information
* **Source:** Seoul Air Pollution Dataset (Item code 9).
* **Target:** PM2.5 Concentration.
* **Note:** Due to the dataset size (~119MB), the raw CSV is excluded from this repository to adhere to GitHub's file size limits. To run the script, please place the dataset in a `/data` folder.

## 🛠️ The Spiral Approach
The algorithm design is implemented in two distinct phases:
1. **Iteration 1 (Baseline):** Simple MLP and Single-layer LSTM to establish initial R² benchmarks and verify temporal data structure.
2. **Iteration 2 (Advanced):** Refined implementation using **Stacked LSTMs**, **Dropout (0.3)**, and **L2 Regularization** to mitigate overfitting and improve prediction accuracy.

## ⚙️ How to Run
1. Ensure you have a Python 3.x environment (e.g., `sensor_env`).
2. Install dependencies:
   ```bash
   pip install pandas numpy tensorflow matplotlib scikit-learn
