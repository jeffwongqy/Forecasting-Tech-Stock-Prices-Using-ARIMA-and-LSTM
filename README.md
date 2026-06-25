# Forecasting Tech Stock Prices Using ARIMA and LSTM

<img width="1372" height="597" alt="Screenshot 2026-06-25 135050" src="https://github.com/user-attachments/assets/59a3a3a3-aca7-4e07-ad74-19dc52a5fb5b" />

# 1. Introduction 
Stock price forecasting is a significant application of time series analysis and machine learning in financial markets, as accurate predictions can support investors and analysts in making informed decisions regarding portfolio management, risk assessment, and trading strategies. This project evaluates and compares two forecasting approaches, namely the Autoregressive Integrated Moving Average (ARIMA) model and the Long Short-Term Memory (LSTM) neural network, using historical stock closing prices of Apple (AAPL), Microsoft (MSFT), and Amazon (AMZN) obtained from Yahoo Finance. To reduce short-term fluctuations, a 5-day moving average was applied before model development. ARIMA serves as a traditional statistical forecasting method based on historical patterns and stationarity assumptions, while LSTM is a deep learning model capable of capturing complex nonlinear relationships in sequential data. The performance of both models was assessed using Root Mean Squared Error (RMSE) and the coefficient of determination (R²) to determine their effectiveness in forecasting technology stock prices.

# 2. Objectives
1. To analyze historical stock price behavior of Apple, Microsoft, and Amazon.
2. To investigate the stationarity characteristics of stock price time series.
3. To develop ARIMA models for statistical forecasting.
4. To develop LSTM neural network models for deep learning-based forecasting.
5. To compare forecasting accuracy between ARIMA and LSTM using R² and RMSE metrics.
6. To evaluate the strengths and limitations of traditional statistical methods and deep learning approaches in stock price prediction.

# 3. Data Splitting
The dataset for each company was divided into:

- Training Set: 80% of observations
- Testing Set: 20% of observations

The training set was used to fit the ARIMA and LSTM models, while the testing set was reserved for evaluating forecasting performance on unseen data.

The chronological order of observations was maintained to preserve the time-dependent structure of the stock price series.
