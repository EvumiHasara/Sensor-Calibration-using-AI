# AI-Driven Calibration of Low-Cost Air Quality Sensors Using MLP and LSTM

This repository contains my Master's project on calibrating low-cost PM2.5 sensors using neural networks. I implemented a spiral development approach, starting with simple MLP and LSTM baselines and progressively adding regularization, dropout, and early stopping in the second iteration.

## 📖 What This Project Is About

Low-cost air quality sensors are affordable and easy to deploy, but they suffer from sensor drift and environmental noise. The goal of this project was to train machine learning models that can map noisy drift-sensor readings to accurate reference-grade measurements.

I used the Seoul Air Pollution dataset and built two types of neural networks:
- **MLP (Multi-Layer Perceptron):** Treats the input as a flat feature vector
- **LSTM (Long Short-Term Memory):** Preserves the temporal structure of the 12-hour lookback window

## 🔄 The Spiral Approach

I structured the work in two iterations:

**Iteration 1 (Baseline)**
- MLP: 1 hidden layer (32 units)
- LSTM: 1 layer (32 units)
- 50 epochs, no regularization
- Goal: Establish baseline performance and check if temporal modeling helps

**Iteration 2 (Advanced)**
- MLP: 2 layers (128 → 64) + L2 regularization + Dropout(0.3)
- LSTM: Stacked layers (64 → 32) + Dropout(0.2)
- Up to 100 epochs with EarlyStopping (patience=10)
- Goal: Reduce overfitting and see if increased capacity improves calibration

## 📊 Results

| Model | R² Score | RMSE (μg/m³) | Training Time |
|-------|----------|--------------|---------------|
| MLP (Iter 1) | 0.0298 | 14.42 | ~5 min |
| LSTM (Iter 1) | 0.0129 | 14.54 | ~8 min |
| Deep MLP (Iter 2) | -0.0290 | 14.85 | ~12 min |
| Deep LSTM (Iter 2) | -0.0269 | 14.83 | ~15 min |
| **Target** | **≥ 0.70** | **< 15** | **< 30 min** |

**What this means:**
- RMSE is close to the target, but R² is near zero (or negative)
- The models learned to predict a constant value near the dataset mean (~18 μg/m³)
- Iteration 2 actually performed worse, which pointed to over-regularization and weak input signal

## 🛠️ How to Run It

1. **Clone the repo**
   ```bash
   git clone https://github.com/EvumiHasara/Sensor-Calibration-using-AI.git
   cd Sensor-Calibration-using-AI
