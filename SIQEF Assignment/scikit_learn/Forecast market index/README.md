# Forecast market index

## Introduction
This project applies differnent ML method(SVM, RandomForest, Naive Bayes) to analyze and predict the market index (399300.XSHE) time series data.

## Data description
We use 'talib' package to extract the characters of this time series data.We choose [SMA，WMA，MOM，STCK，STCD，MACD，RSI，WILLR，CCI，MFI，OBV，ROC，CMO] technical indices as the features to train our model.

## Reference
https://www.joinquant.com/view/community/detail/e1ecb4e2ab137e90707b2877debe6e45
