# Predict Happiness

## Introduction
This is a competition question from Tianchi, aliyun.

### Competition background
In the field of social sciences, the study of happiness occupies an important position. This topic involving philosophy, psychology, sociology, economics and other disciplines is complicated and interesting; at the same time, it is closely related to everyone's life, and everyone has their own standards for measuring happiness. If we can find the commonality that affects happiness, will there be more fun in life? If we can find policy factors that affect happiness, we can optimize the allocation of resources to improve the happiness of the people. At present, social science research focuses on the interpretability of variables and the implementation of future policies, mainly using linear regression and logistic regression methods, focusing on economic and demographic factors such as income, health, occupation, social relations, and leisure; as well as government public services, macro There have been a series of speculations and discoveries on macro factors such as the economic environment and tax burden.

The competition question tried the classic subject of happiness prediction, hoping to try algorithms in other dimensions beyond the existing social science research, combining the advantages of multiple disciplines, mining potential influencing factors, and discovering more interpretable and understandable correlations. .

### Question description
The contest questions use the results of a questionnaire survey of public data, and select multiple sets of variables, including individual variables (gender, age, region, occupation, health, marriage and political status, etc.), family variables (parents, spouses, children, family capital, etc.), social attitudes (fairness, credit, public services, etc.) to predict their evaluation of happiness.

The accuracy of well-being prediction is not the only purpose of the question, but I hope that the contestants will explore and gain some insights into the relationship between variables and the meaning of variable groups.


## Data description

### Data types and files
Considering the large number of variables and the complicated relationships among some variables, the data is divided into two types: full version and condensed version. You can start with the condensed version and use the full version to dig out more information after familiarizing yourself with the competition questions. The complete file is the full version of the variable data, and the abbr file is the simplified version of the variable.

The index file contains the questionnaire question corresponding to each variable and the meaning of the value of the variable.

The survey file is the original questionnaire of the data source, as a supplement to facilitate the understanding of the question background.

### Data source:
 The data used in the question comes from the "China Comprehensive Social Survey (CGSS)" project hosted by the China Survey and Data Center of Renmin University of China. Thanks to this organization and its staff for providing data assistance. The China Comprehensive Social Survey is a cross-sectional survey of multi-stage stratified sampling.

 The competition questions are based on data mining and analysis, and the use of external data is not restricted, such as public data such as macroeconomic indicators and government redistribution policies. Contestants are welcome to exchange and share.

## Evaluation criteria
score = \frac{1}{n}\sum_{1}^{n}(y_i-y^*)^2score= 
n
1
​	
  
1
∑
n
​	
 (y 
i
​	
 −y 
∗
 ) 
2
 


## Reference
https://tianchi.aliyun.com/competition/entrance/231702/introduction?spm=5176.12281973.1005.9.3dd52448YzhqH9