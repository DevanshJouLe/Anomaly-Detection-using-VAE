# Anomaly Detection using VAE and Re-Encoder LSTM

### Deep Learning Course Project: Prof. Sumohana Channappayya (IIT Hyderabad)  
**Timeline:** September 2025  

---

## Overview
This project implements a **Variational Autoencoder (VAE)** combined with an **LSTM-based architecture** for unsupervised anomaly detection in sequential data.  
A **Re-Encoder module** is introduced to enforce latent-space consistency, improving robustness against noise and distributional drift in time-series inputs to enhance reconstruction reliability.

---

## Key Features
- **LSTM-based VAE** trained exclusively on normal sequences to model temporal dependencies.  
- **Re-Encoder module** for latent consistency between original and reconstructed data representations.  
- **Extended ELBO loss** combining reconstruction, dual KL-divergence, and latent consistency penalties.  
- **Statistical residual analysis** using the **Lilliefors test** for anomaly scoring in noisy environments.  
- **Preprocessing pipeline** including Butterworth filtering and adaptive windowing for stable performance.

---

## Results
| Dataset | AUC Score | Description |
|----------|------------|--------------|
| ECG5000 | **0.997** | Clean physiological signals |
| Green Mobility | **0.73** | Noisy multivariate sensor data |

---

## Architecture
Input Sequence → Encoder (LSTM) → Latent (μ, σ)
          ↓
      Reparameterization
          ↓
      Decoder (LSTM)
          ↓
   Re-Encoder (LSTM) → Latent′ (μ′, σ′)

###Loss = Reconstruction + KL₁ + KL₂ + Latent Consistency
