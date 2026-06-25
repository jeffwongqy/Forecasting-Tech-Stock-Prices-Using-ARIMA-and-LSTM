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

#### 4.1.2 Partial Autocorrelation Function (PACF)

PACF plots were used to identify the number of significant autoregressive terms. Significant spikes in the initial lags suggested the presence of autoregressive components.

#### 4.1.3 Augmented Dickey-Fuller (ADF) Test

The ADF test was performed on the training data for Apple, Microsoft, and Amazon.

Hypotheses:

H₀: The time series is non-stationary.
H₁: The time series is stationary.

The initial ADF results showed insufficient evidence to reject the null hypothesis, indicating that the original stock price series were non-stationary and required differencing.


### 4.2 First Differencing - PACF, ACF, and ADFuller

To remove trend effects and stabilize the mean, first-order differencing was applied: Y't = Y_t - Y{t-1}

The differenced series were then re-evaluated.

(i) ACF Analysis

After differencing, the autocorrelation structure decayed more rapidly, suggesting improved stationarity.

(ii) PACF Analysis

The PACF plots showed fewer significant spikes, indicating that the differenced series could be adequately modeled using a small number of AR terms.

(iii) ADF Test Results

The ADF test was repeated on the differenced data. The resulting p-values became significantly smaller, providing evidence that the transformed series were stationary.

Therefore, a differencing order of: d = 1 was selected for all ARIMA models.

### 4.3 ARIMA Training 
Based on the stationarity analysis and ACF/PACF observations, the following ARIMA configurations were selected:

TBC

The models were trained using the training datasets.

Model summaries were examined to evaluate:

Parameter significance
Residual variance
Information criteria (AIC and BIC)
Overall model adequacy

For forecasting, a rolling forecasting strategy was implemented. After each prediction, the actual observation was appended to the historical dataset before generating the next forecast. This approach mimics real-world forecasting conditions and improves prediction reliability.


### 4.4 ARIMA Evaluation
#### 4.4.1 Diagnostics Plot
Diagnostic plots were generated using the fitted ARIMA models.

The diagnostics included:

Standardized residuals.
Histogram and density estimate of residuals.
Normal Q-Q plot.
Correlogram of residuals.

The residuals were expected to resemble white noise, indicating that the model successfully captured the underlying structure of the stock price series.

#### 4.4.2 Performance Metrics 
Two evaluation metrics were used:

(1) Root Mean Squared Error (RMSE)

RMSE measures average prediction error magnitude.

Coefficient of Determination (R²)

R² measures how well the model explains the variability in stock prices.

The ARIMA forecasts were generated on the testing dataset and evaluated using RMSE and R². Higher R² values and lower RMSE values indicate better forecasting performance.

