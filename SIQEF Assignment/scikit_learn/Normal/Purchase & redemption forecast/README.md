# Predict Happiness

## Introduction
This is a competition question from Tianchi, aliyun.

Ant Financial has hundreds of millions of members and its business scenarios involve a large amount of capital inflows and outflows every day. Faced with such a large user base, the pressure on capital management will be very high. In the case of ensuring the minimum liquidity risk of funds and meeting daily business operations, it is particularly important to accurately predict the inflow and outflow of funds. The topic of this competition is "Fund Inflow and Outflow Forecast". It is expected that participants can accurately predict the future daily inflow and outflow of funds through the grasp of the purchase and redemption data of Yu'ebao users. For currency funds, capital inflow means subscription behavior, and capital outflow is a redemption behavior.

## Data description

### Data types and files
The data used in the competition mainly consists of four parts, namely user basic information data, user subscription and redemption data, return rate table and inter-bank lending rate table. Four sets of data are introduced below.

1. User Information Form
User information table: user_profile_table. We randomly selected about 30,000 users in total, some of which appeared for the first time in September 2014, and these users are only in the test data. Therefore, the user information table is the basic data of about 28,000 users. After processing on the basis of the original data, it mainly contains the user's gender, city, and constellation. The specific labels are shown in Table 1 below:

 Table 1： User Information Table

|Column name |Type |Meaning |Example    |
|:---------------:|:---------:|:----------------------------------:|:--------------:|
|user_id    |bigint     |User ID    |1234   |
|Sex    |bigint     |User gender (1 male, 0 female)  |0  |
|City   |bigint     |User City   |6081949    |
|constellation      |string     |constellation       |射手座(Sagittarius) |

* Note: For the example of the data, we show English in brackets for reader to understand; However, in the real dataset there is no English translation. 

2. User Subscription and Redemption Data Sheet
User subscription and redemption data table: user_balance_table.
There are purchase and redemption information from 20130701 to 20140831, as well as all sub-category information, and the data has been desensitized.
The data after desensitization basically maintained the original data trend. The data mainly includes user operation time and operation records. The operation record includes two parts: purchase and redemption. The unit of the amount is cents, which is 0.01 yuan. If the total consumption of the user today is 0, that is, consume_amt=0, the four word categories are empty.

 Table 2: User subscription and redemption data

|Column name   |Type   |Meaning   |Example   |
|:--------------------:|:----------:|:--------------------------------------------:|:-----------------:|
|user_id    |bigint |User id    |1234   |
|report_date    |string |Date   |20140407   |
|tBalance   |bigint |Today's balance   |109004 |
|yBalance   |bigint |Yesterday's balance   |97389  |
|total_purchase_amt |bigint |Today's total purchases = direct purchases + revenue |21876  |
|direct_purchase_amt    |bigint |Direct purchases today |21863  |
|purchase_bal_amt   |bigint |Alipay balance purchases today   |0  |
|purchase_bank_amt  |bigint |Bank card purchases today   |21863  |
|total_redeem_amt   |bigint |Today's total redemption volume = consumption + transfer out |10261  |
|consume_amt    |bigint |Total consumption today   |0  |
|transfer_amt   |bigint |Total transfer out today   |10261  |
|tftobal_amt    |bigint |Total balance transferred out to Alipay today   |0  |
|tftocard_amt   |bigint |Total number of transfers to bank cards today   |10261  |
|share_amt  |bigint |Today's earnings   |13 |
|category1  |bigint |Total consumption of category 1 today   |0  |
|category2  |bigint |Total consumption of category 2 today   |0  |
|category3  |bigint |Total consumption of category 3 today   |0  |
|category4  |bigint |Total consumption of category 4 today   |0  |

* Note 1: The above-mentioned data are all desensitized, and the income is obtained by recalculation. The calculation method is processed according to the simplified calculation method. The specific calculation method is described in the next section Yu'ebao income calculation method.

* Note 2 : The desensitized data guarantees that today's balance = yesterday's balance + today's purchase-today's redemption. There will be no negative value.

3. Yield table
The yield table is Yu'ebao's income rate table within 14 months: mfd_day_share_interest. The specific fields are shown in Table 3

 Table 3: Yield table

|Column name   |Type   |Meaning   |Example   |
|:--------------------:|:----------:|:---------------------------------------------:|:-------------:|
|mfd_date   |string     |日期   |20140102
|mfd_daily_yield    |double     |Ten thousand shares of income, ie, income of 10,000 yuan  |1.5787 |
|mfd_7daily_yield   |double     |Seven-day annualized rate of return (%)     |6.307  |


4. Shanghai Interbank Offered Rate(Shibor) Table
The inter-bank lending rate table is the inter-bank lending rate during a 14-month period (all annualized rates): mfd_bank_shibor. Details are shown in Table 4 below:
 Table 4 : Shibor Table

|Column name   |Type   |Meaning   |Example   |
|:----------------:|:----------:|:-------------------:|:---------:|
|mfd_date   |String |Date   |20140102   |  
|Interest_O_N   |Double |Overnight interest rate (%)  |2.8    |   
|Interest_1_W   |Double |1-week interest rate (%)  |4.25   |
|Interest_2_W   |Double |2-week interest rate (%)  |4.9    |
|Interest_1_M   |Double |1-month interest rate (%)  |5.04   |
|Interest_3_M   |Double |3-month interest rate (%)  |4.91   |
|Interest_6_M   |Double |6-month interest rate (%)  |4.79   |
|Interest_9_M   |Double |9-month interest rate (%)  |4.76   |
|Interest_1_Y   |Double |1-year interest rate (%)   |4.78   |


5. Income calculation method

The income method of Yu'ebao in this contest is mainly based on the actual Yu'ebao income calculation method, but has been simplified to a certain extent. The simplified calculation method here is as follows:

First of all, the time for calculating the income is no longer an accounting day, but a natural day, separated by 0 o’clock. If it is the amount transferred in or out before 0 o’clock, it is counted as yesterday, if it is transferred in or out after 0 o’clock. The amount is counted as today's.

Then, the display time of the income, that is, the time when the first income is actually credited to the user's account, is calculated in the form of the following table. Take the display from Monday to Wednesday as an example: if a user deposits 10,000 yuan on Monday, then the amount is confirmed on Monday, and revenue starts to be generated on Tuesday, and the user’s balance is still 10,000 yuan. On Wednesday, the revenue generated on Tuesday will be into the user’s account, the user’s account now shows 1,0001.1 yuan. Other calculations are based on the rules in the table.

 Table 5: Simplified Yu'e Bao income calculation rule table

|Transfer-in time   |Time of first showing earnings   |
|:-------------------:|:-------------------------:|
|Monday |Wednesday |
|Tuesday |Thursday |
|Wednesday | Friday |
|Thursday | Saturday |
|Friday |Next Tuesday |
|Saturday |Next Wednesday |
|Weekday |Next Wednesday |


6. The result form that the contestant needs to submit:

 Table 6 The result table submitted by the contestants: tc_comp_predict_table

|Label |Type |Meaning |Example |
|:----------------:|:----------:|:--------------:|:-------------:|
|report_date    |bigint |Date   |20140901   |
|purchase   |bigint |Total subscription   |40000000   |
|redeem |bigint |Total redemption   |30000000   |

Each row of data is the predicted value of the total subscription and redemption for one day. There are 30 rows of data for one row per day in September 2014. Purchase and redeem are both amount data, accurate to the cent, not to the yuan.
        The scoring data format is required to be consistent with the "sample file of player results data". The result table is named: tc_comp_predict_table. The fields are separated by commas. 

## Evaluation criteria

The design of the evaluation index mainly expects players to predict the total amount of purchase and redemption data for each day in the next 30 days as accurately as possible, while taking into account the various situations that may exist. For example, some players have very accurate predictions for 29 days out of 30 days, but the prediction results on a certain day may have a large error, while some players’ predictions every day for 30 days are not very accurate and the error is large. If absolute error is used, the performance of the former is worse than the latter, but in actual business people may be more inclined to the former. Therefore, the integral calculation method is finally selected: the daily error is calculated by the relative error, and then according to the relative error of the user's predicted purchase and redemption, a score of the daily forecast result is obtained through the score function mapping, and the scores within 30 days are summarized, combining with the actual business tendency, the weighted sum of the predicted scores of the total purchase and redemption is performed to obtain the final score. The specific operations are as follows:

1) Calculate the proportion of the error between the total daily subscription and redemption of all users on the test set and the total actual value in the true value: Purchase_i and Redeem_i。


2) The purchase prediction score is related to Purchase_i, and the redemption prediction score is related to Redeem_i. The calculation formula between the error and the score is not announced, but it is guaranteed that the calculation formula is monotonically decreasing, that is, the smaller the error, the higher the score, and the greater the error, the score The lower. When the purchase error Purchase_i=0 on the i-th day, the score for that day is 10 points; when Purchasei> 0.3, the score is 0.

3) Finally announced total points = purchase prediction score *45% + redemption prediction score *55%. 


## Reference
https://tianchi.aliyun.com/competition/entrance/231573/information