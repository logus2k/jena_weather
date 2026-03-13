# Introduction

Time series forecasting plays a central role in meteorology, environmental monitoring, and climate-aware decision-making. The availability of long-term, high-resolution atmospheric measurements enables the development of deep learning models capable of capturing complex temporal dependencies and nonlinear relationships in environmental systems.

This project focuses on the Jena Climate dataset, a multivariate meteorological time series collected between 2009 and 2016 at the Max Planck Institute for Biogeochemistry. The dataset contains 14 atmospheric variables recorded every 10 minutes, including temperature, pressure, humidity, and wind-related measurements. 

However, in accordance with the project guidelines, only a subset of variables is used in the experiments in order to avoid near-deterministic relationships with the target variable. The input variables considered in this study are: past air temperature (T (degC)), atmospheric pressure (p (mbar)), relative humidity (rh (%)), wind speed (wv (m/s)), maximum wind speed (max. wv (m/s)), and wind direction (wd (deg)), together with the corresponding timestamp.

Following the project guidelines, the forecasting task is formulated as a multivariate-input, univariate-output, multi-step forecasting problem, where past meteorological observations are used to predict the future air temperature over a fixed horizon of 24 hours. The input variables include past air temperature together with atmospheric pressure, relative humidity, wind speed, maximum wind speed, and wind direction.

To address this problem, we implement and evaluate deep learning models designed for sequential data, namely Gated Recurrent Units (GRU) and Transformer-based architectures. The study follows a complete time series modelling pipeline including exploratory data analysis, data preprocessing, temporal resampling, windowing strategies, model development, hyperparameter exploration, and performance evaluation.

The final objective is not only to obtain accurate forecasts, but also to understand how different architectures and modelling choices influence forecasting performance, particularly across different forecast horizons.