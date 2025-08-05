<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/><img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"/> <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white"/> <img src="https://img.shields.io/badge/numpy-013243?style=flat-square&logo=numpy&logoColor=white"/> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=TensorFlow&logoColor=white"/> <img src = "https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>

# Predicting Stock Price Index using the Wavelet Transform and Temporal Fusion Transformer Model

- Model Workflow
  
  <img width="192" height="304" alt="image" src="https://github.com/user-attachments/assets/0f389be6-d251-4f87-83e5-4f2619f68b64" />
  
- Test Result
  
  <img width="390" height="283" alt="image" src="https://github.com/user-attachments/assets/0422a2ae-813e-4126-ad07-909f2f8dea63" />

# Overview
1. Transformer Model for Paper Submission to the ACK2024 Conference
2. For reduce noise of Stock price data, preprocessing the data with Wavelet Transform.
3. Transformer models were used to improve errors that occur when predicting long-term time series data.
4. Transformer models extract the points in time series that are relevant for prediction through attention, enabling more accurate forecasts.

# Code Description

Models - LSTM, Temporal Fusion Transformer and LightGBM Model for Predicting Stock Price

param_tuning - Parameter Tuning for LightGBM

# Paper Link
https://www.manuscriptlink.com/society/kips/conference/ack2024/file/downloadSoConfManuscript/abs/KIPS_C2024B0077

# Cite
This code has been implemented based on PyTorch Forecasting's TemporalFusionTransformer.

https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer._tft.TemporalFusionTransformer.html
