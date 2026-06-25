# Forecasting Tech Stock Prices Using ARIMA and LSTM

<img width="1372" height="597" alt="Screenshot 2026-06-25 135050" src="https://github.com/user-attachments/assets/59a3a3a3-aca7-4e07-ad74-19dc52a5fb5b" />

## 1. Introduction 
Stock price forecasting is a significant application of time series analysis and machine learning in financial markets, as accurate predictions can support investors and analysts in making informed decisions regarding portfolio management, risk assessment, and trading strategies. This project evaluates and compares two forecasting approaches, namely the Autoregressive Integrated Moving Average (ARIMA) model and the Long Short-Term Memory (LSTM) neural network, using historical stock closing prices of Apple (AAPL), Microsoft (MSFT), and Amazon (AMZN) obtained from Yahoo Finance. To reduce short-term fluctuations, a 5-day moving average was applied before model development. ARIMA serves as a traditional statistical forecasting method based on historical patterns and stationarity assumptions, while LSTM is a deep learning model capable of capturing complex nonlinear relationships in sequential data. The performance of both models was assessed using Root Mean Squared Error (RMSE) and the coefficient of determination (R²) to determine their effectiveness in forecasting technology stock prices.

## 2. Objectives
1. To analyze historical stock price behavior of Apple, Microsoft, and Amazon.
2. To investigate the stationarity characteristics of stock price time series.
3. To develop ARIMA models for statistical forecasting.
4. To develop LSTM neural network models for deep learning-based forecasting.
5. To compare forecasting accuracy between ARIMA and LSTM using R² and RMSE metrics.
6. To evaluate the strengths and limitations of traditional statistical methods and deep learning approaches in stock price prediction.

## 3. Data Splitting
The dataset for each company was divided into:

- Training Set: 80% of observations
- Testing Set: 20% of observations

The training set was used to fit the ARIMA and LSTM models, while the testing set was reserved for evaluating forecasting performance on unseen data.

The chronological order of observations was maintained to preserve the time-dependent structure of the stock price series.

## 4. ARIMA Model
ARIMA (Autoregressive Integrated Moving Average) is a classical forecasting model designed for univariate time series data. It combines:

AR (Autoregressive): dependence on previous observations.
I (Integrated): differencing to achieve stationarity.
MA (Moving Average): dependence on previous forecast errors.

The general ARIMA model is represented as: ARIMA(p,d,q), where: (p) = autoregressive order, (d) = differencing order, (q) = moving average order.

### 4.1 Stationarity Test - PACF, ACF, and ADFuller
Before fitting ARIMA models, the stock price series were examined for stationarity.

<img width="1260" height="580" alt="aapl" src="https://github.com/user-attachments/assets/22ed8125-4a37-4739-a699-b8a7c124a21a" />

<img width="1260" height="580" alt="msft" src="https://github.com/user-attachments/assets/18102929-24c4-475c-a50c-33c7004d5841" />

<img width="1260" height="580" alt="amzn" src="https://github.com/user-attachments/assets/175fa77b-23aa-4e49-b693-25a93a20ef4d" />

#### 4.1.1 Autocorrelation Function (ACF)
The ACF plots were generated to identify serial dependencies across lagged observations. The slow decay observed in the ACF plots indicated strong autocorrelation and suggested non-stationary behavior.

<img width="850" height="550" alt="aapl_acf" src="https://github.com/user-attachments/assets/0e3fe490-0a71-4eb7-8f36-8b50da13c28c" />

<img width="850" height="550" alt="msft_acf" src="https://github.com/user-attachments/assets/3843e944-d3e7-4195-9f08-e363eb443708" />

<img width="850" height="550" alt="amzn_acf" src="https://github.com/user-attachments/assets/39a28112-c9fd-40a8-929c-ac552db5b62a" />

#### 4.1.2 Partial Autocorrelation Function (PACF)
PACF plots were used to identify the number of significant autoregressive terms. Significant spikes in the initial lags suggested the presence of autoregressive components.

<img width="850" height="550" alt="aapl_pacf" src="https://github.com/user-attachments/assets/257dc7aa-197d-44e4-81e8-82f688aaada1" />

<img width="850" height="550" alt="msft_pacf" src="https://github.com/user-attachments/assets/6d949cdd-d544-413b-b1e3-3534acc2fd9c" />

<img width="850" height="550" alt="amzn_pacf" src="https://github.com/user-attachments/assets/9edb2900-3acd-4b6b-aa80-901050e6d3a8" />

#### 4.1.3 Augmented Dickey-Fuller (ADF) Test
The ADF test was performed on the training data for Apple, Microsoft, and Amazon.

Hypotheses:

H₀: The time series is non-stationary.

H₁: The time series is stationary.

**Apple:**

<img width="225" height="50" alt="aapl_adfuller" src="https://github.com/user-attachments/assets/972dde78-6cf0-4ee6-ba04-c2a257892b3e" />

**Microsoft:**

<img width="225" height="50" alt="msft_adfuller" src="https://github.com/user-attachments/assets/2641239e-15d6-4e50-bf4a-81ff56833680" />

**Amazon:**

<img width="225" height="50" alt="amzn_adfuller" src="https://github.com/user-attachments/assets/de13604a-1444-4dd2-a4ef-0a5884bf23d3" />

The initial ADF results showed insufficient evidence to reject the null hypothesis, indicating that the original stock price series were non-stationary and required differencing.


### 4.2 First Differencing - PACF, ACF, and ADFuller
To remove trend effects and stabilize the mean, first-order differencing was applied: Y't = Y_t - Y{t-1}

<img width="1260" height="580" alt="aapl" src="https://github.com/user-attachments/assets/37d84724-81be-40f8-b4b3-6741a97e24e2" />

<img width="1260" height="580" alt="msft" src="https://github.com/user-attachments/assets/6e6cc6bd-cb6f-40d9-89c6-b76aea09015c" />

<img width="1260" height="580" alt="amzn" src="https://github.com/user-attachments/assets/f28cd888-8d72-4c94-8c6b-c787928b8500" />

The differenced series were then re-evaluated.


#### 4.2.1 ACF Analysis

After differencing, the autocorrelation structure decayed more rapidly, suggesting improved stationarity.

<img width="850" height="550" alt="aapl_acf" src="https://github.com/user-attachments/assets/3fe05075-81c8-4d79-bae7-374f57bfac4b" />

<img width="850" height="550" alt="msft_acf" src="https://github.com/user-attachments/assets/e8729e63-6652-41c3-9086-8b731fd805d1" />

<img width="850" height="550" alt="amzn_acf" src="https://github.com/user-attachments/assets/ef37294e-bea0-40bc-aff3-af69ed46dc3c" />


#### 4.2.2 PACF Analysis
The PACF plots showed fewer significant spikes, indicating that the differenced series could be adequately modeled using a small number of AR terms.

<img width="850" height="550" alt="aapl_pacf" src="https://github.com/user-attachments/assets/0463e7b8-4125-4d66-8046-3f4037a8fef7" />

<img width="850" height="550" alt="msft_pacf" src="https://github.com/user-attachments/assets/4ea52a29-76e6-4756-9703-8305c8a6dc63" />

<img width="850" height="550" alt="amzn_pacf" src="https://github.com/user-attachments/assets/79b99544-e8f2-41cc-bd79-5d127991fddb" />

#### 4.2.3 ADF Test Results
The ADF test was repeated on the differenced data. The resulting p-values became significantly smaller, providing evidence that the transformed series were stationary.

**Apple:**

<img width="225" height="50" alt="aapl_adfuller" src="https://github.com/user-attachments/assets/e41108d3-1214-433d-af17-1031a66c120e" />

**Microsoft:**

<img width="225" height="50" alt="msft_adfuller" src="https://github.com/user-attachments/assets/2c5ee802-534e-4042-814f-0db150e3d9b9" />

**Amazon:**

<img width="225" height="50" alt="amzn_adfuller" src="https://github.com/user-attachments/assets/8e36800b-6c58-4944-a0ac-63df6c5d340e" />

Therefore, a differencing order of: d = 1 was selected for all ARIMA models.


### 4.3 ARIMA Training 
Based on the stationarity analysis and ACF/PACF observations, the following ARIMA configurations were selected. The models were trained using the training datasets.

<img width="750" height="485" alt="aapl" src="https://github.com/user-attachments/assets/9d23a0fc-6703-4403-87cd-1fab2464d278" />

<img width="750" height="485" alt="msft" src="https://github.com/user-attachments/assets/b59fbee4-3d62-4ed6-8baa-e8ddf260ce22" />

<img width="750" height="485" alt="amazon" src="https://github.com/user-attachments/assets/8521a16e-46c6-4581-9159-838e299aa2a5" />


### 4.4 ARIMA Evaluation
#### 4.4.1 Diagnostics Plot
Diagnostic plots were generated using the fitted ARIMA models.

**Apple:**
<img width="989" height="790" alt="aapl_plot" src="https://github.com/user-attachments/assets/c67c5474-fc27-4f87-b518-1fc7c0b377f1" />

**Microsoft:**
<img width="989" height="790" alt="msft_plot" src="https://github.com/user-attachments/assets/79bc4f52-ba04-4522-b148-ed01eb52b3ea" />

**Amazon:**
<img width="989" height="790" alt="amzn_plot" src="https://github.com/user-attachments/assets/ad98c2fe-457f-47b2-ab67-55cc4cb286b0" />



The residuals were expected to resemble white noise, indicating that the model successfully captured the underlying structure of the stock price series.

#### 4.4.2 Performance Metrics 
Two evaluation metrics were used:

(1) Root Mean Squared Error (RMSE)

RMSE measures average prediction error magnitude.

Coefficient of Determination (R²)

R² measures how well the model explains the variability in stock prices.

The ARIMA forecasts were generated on the testing dataset and evaluated using RMSE and R². Higher R² values and lower RMSE values indicate better forecasting performance.

