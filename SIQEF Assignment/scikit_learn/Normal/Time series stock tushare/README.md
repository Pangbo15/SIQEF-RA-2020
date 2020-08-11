# Tushare stock time series analysis


## Data descrption
This project is for time series analysis and forecasting of '雪人股份' `002639.SZ` stocks, and the data set is among the whole year of 2017. We use tushare API for free data access. The characteristics of this data set are shown in the following table:

|Feature | Explanation|
| :------------: |:---------------:|
|date  | Date  |
|open  | Opening price  |
|high  | Highest price  |
|close  | Closing price  |
|low  | Lowest price  |
|volume  | Stock trading volume  |
|price_change  | Price change  |
|p_change  | Percentage change  |
|ma5  | 5-day moving average  |
|ma10  | 10-day moving average  |
|ma20  | 20-day moving average  |
|v_ma5  | volume:5-day moving average  |
|v_ma10  | volume:10-day moving average  |
|v_ma20  | volume:20-day moving average  |

First, we conduct feature exploration and use the `Dickey-Fuller test` to evaluate the stationarity of the time series. Then, we make the time series stationary through logarithm and difference operations. Then we calculate the autocorrelation of the time series through `Durbin Watson Statistics`. Finally, we use `ARIMA` for time series modeling and analysis, and the final **MSE: 0.2393**


## Research
https://github.com/wzy6642/Machine-Learning-Case-Studies
