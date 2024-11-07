### DEVELOPED BY : GOWRISANKAR P
### REGISTER NO : 212222230041
# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 

### AIM:
To Create a project on Time series analysis on MentalHealthSurvey using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of mental Health 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
file_path = 'MentalHealthSurvey.csv'
data = pd.read_csv(file_path)
series = data['depression']
result = adfuller(series)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
plt.figure(figsize=(12, 6))
plt.subplot(311)
plt.plot(series)
plt.title("Depression Series Plot")

plt.subplot(312)
plot_acf(series, ax=plt.gca(), lags=20)
plt.title("ACF Plot")

plt.subplot(313)
plot_pacf(series, ax=plt.gca(), lags=20)
plt.title("PACF Plot")
plt.tight_layout()
plt.show()

if result[1] > 0.05:  # If p-value > 0.05, series is not stationary
    series_diff = series.diff().dropna()
else:
    series_diff = series

model = ARIMA(series_diff, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

forecast = model_fit.forecast(steps=10)
print("Forecasted values:", forecast)

plt.figure(figsize=(10, 5))
plt.title("ARIMA MODEL")
plt.plot(series, label="Original Series")
plt.plot(range(len(series), len(series) + len(forecast)), forecast, label="Forecast", color='red')
plt.legend()
plt.show()

```
### OUTPUT:
![image](https://github.com/user-attachments/assets/5d460de2-f659-4f37-a433-05c0400cdbbb)
![image](https://github.com/user-attachments/assets/3cbd3ea9-7ff0-4137-9ad0-227a6726b001)
![image](https://github.com/user-attachments/assets/450e9ec7-0216-452d-8621-1199bea4f5fa)
![image](https://github.com/user-attachments/assets/0c3e6c48-f024-4ab4-b54a-5a741eae8829)
### RESULT:
Thus the program run successfully based on the ARIMA model using python.
