# Project Title: Store Sales Prediction based on time series data
## Problem Statement:
Predicting the sale of different types of products sold in Favorita stores (Ecuador), by considoring factors like historical time series sales data, promotions, holiday seasons and oil prices.

## Professor: 
Carlos Rojas

## Term: 
Fall 2022

## Team Number: 
10

## Team Members:
- Hardy Leung (ksleung) 
- Loukya Tammineni (LoukyaTammineni)
- Suhas Anand Balagar (suhasAB)
- Xichang Yu (Codyyu36)

## 1.Abstract:

The ability to predict the sales of a variety of stores is highly sought out in supply chain logistics, as it finds applications in increasing customer satisfaction and reducing food waste. <br>
We are proposing use of multiple **Supervised Learning methods** to predict the sales of stores based on time series dataset of Corporación Favorita, an Ecuador based grocery retailer. Ecuador is a country whose economy is strongly dependent on the oil and fluctuates with the price of oil. <br>
We are planning to use dataset from an ongoing Kaggle competition, “Store Sales - Time Series Forecasting”. The dataset includes multiple csv sheets of time series data. We will try to evaluate the different aspects that might impact the sales in a store like Holiday seasons, Oil prices and historical sales data from a variety of stores. <br> 
The preprocessing of data will include checking for missing values and if found, imputing them so that no data is disregarded. We have also checked for frequency distribution of data elements as part of Data exploration. We have calculated and plotted correlation mapping to establish the correlations between different input and output parameters such as holiday events, oil prices, transactions per store type, dates of salary, natural calamities etc.
<br>
We plan on using different Supervised Learning models to predict the prices and then evaluate which of those models give out the best results. Supervised learning approach works best in this case, as we have huge time series data of both input and the output parameters mentioned above.
Few of the supervised learning options we can consider are <br>
- Linear Regression
- Gradient Boost(XGBoost)
- Random Forest (RF)
- Support Vector Machine(SVM) /Support Vector Regression (SVR)
- Long Short-Term Memory (LSTM)
- Ensemble Regression by combining XGBoost, RF, Ridge and SVR models

<br>
Once the models are trained on training data and developed, we can evaluate the performance of these models on the test dataset.

## Presentation Link: https://drive.google.com/file/d/11LG8q3dlsKGfcd4Luza5-uhRGnc1-NsB/view?usp=sharing

