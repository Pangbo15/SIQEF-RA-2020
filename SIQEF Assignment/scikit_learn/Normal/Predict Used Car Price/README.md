# Predict used car price

## Introduction
This is a competition question from Tianchi, aliyun.
* At that case we didn't show how to conduct the real predict model, but rather provide some preprocessing(feature engineering) method, which will be of great help when you want to improve your model accuracy or interpretability. You can try the forecast method in the class among those data.


The task of the contest is to predict the transaction price of used cars. The data comes from the used car transaction records of a trading platform. The total data volume exceeds 40w and contains 31 columns of variable information, 15 of which are anonymous variables (The official data source cannot provide its name due to business confidential). In order to ensure the fairness of the game, 150,000 pieces of data will be extracted as the training set, 50,000 pieces of data will be used as test set A, and 50,000 pieces will be used as test set B. At the same time, information such as `name`, `model`, `brand`, and `regionCode` will be desensitized.


## Data Description：

* Note: the `test` and `train` datasets in the file are the original data. `data_for_lr` and `data_for_tree` csv files are the product after our feature engineering (but you can also use it to run logistic_regression and decision_tree accordingly).

|Field  |Description    |
|:-------------:|:-----------------------------------------------------------------:|
|SaleID |	Sale ID, the only id index in this project    |
|name   |	Automobile transaction name, desensitized    |
|regDate    |	Car registration date, such as 20160101 = January 01, 2016  |
|model  |	Model code, desensitized    |
|brand  |	Car brand, desensitized    |
|bodyType   |	Body Type: Luxury Car: 0, Mini Car-1, Van-2, Bus-3, Convertible-4, Two-door Car-5, Commercial Car-6, Mixer-7   |
|fuelType   |	Fuel type: gasoline-0, diesel-1, liquefied petroleum gas-2, natural gas-3, hybrid-4, other-5, electric-6  |
|gearbox    |	Transmission: manual-0, automatic-1 |
|power  |	Engine power: among [0, 600] |
|kilometer  |	The distance traveled by the car, measured in 10,000 km   |
|notRepairedDamage  | Whether the car has unrepaired damage: yes-0, no-1  |
|regionCode |	Area code, desensitized  |
|seller |	Seller: Individual-0, Non-individual-1  |
|offerType  |	Quotation type: offer-0, request-1  |
|creatDate  |	The time when the car goes online and starts to sell    |
|price  |	Used car price (our y variable)***  |
|v系列特征(v series features)  |	Anonymous features, 15 anonymous features including v0-14 |


## Evaluation criteria
The evaluation standard is MAE (Mean Absolute Error).
The smaller the MAE, the more accurate the model predicts.


## Reference
https://tianchi.aliyun.com/competition/entrance/231784/information