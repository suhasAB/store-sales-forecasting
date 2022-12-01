
---
title: LiBerTY -- Store Sales Forcast with Machine Learning
date: "November 2022"
author:
  - Suhas Anand Balagar^[San José State University, \texttt{suhasAB@github}]
  - Hardy Leung^[San José State University, \texttt{ksleung@github}]
  - Loukya Tammineni^[San José State University, \texttt{LoukyaTammineni@github}]
  - Xichang Yu^[San José State University, \texttt{Codyyu36@github}]

header-includes: |
  \usepackage{booktabs}
  \usepackage{caption}
---

# Abstract

LiBerTY is an ensemble regression engine to predict the store sales
given past sales figures. It employed a variety of robust and general
data analysis and machine learning techniques to achieve good result
within a short amount of time, competitive to existing best-known
result while being robust and generally applicable to other problems.

# Introduction[^1]
[^1]: Suhas's section

The ability to predict the sales of a variety of stores is highly sought
out in supply chain logistics, as it finds applications in increasing
customer satisfaction and reducing food waste.  We are proposing use of
multiple Supervised Learning methods to predict the sales of stores based on
time series dataset of Corporación Favorita, an Ecuador based grocery retailer. Ecuador is a country whose economy is strongly dependent on the oil and
fluctuates with the price of oil.

Our work focused on a dataset from an ongoing Kaggle competition [@kaggle],
``Store Sales -- Time Series Forecasting''. The dataset includes multiple
csv sheets of time series data. We will try to evaluate the different
aspects that might impact the sales in a store like Holiday seasons,
Oil prices and historical sales data across all stores.

We plan on using different Supervised Learning models to predict
the prices and then evaluate which of those models give out the best
results. Supervised learning approach works best in this case, as we
have huge time series data of both input and the output parameters
mentioned above. We applied several transformation and optimization to
improve the data quality in preparation for the optimization.
To seek the best store sales prediction, We have evaluated 
several models including linear regression, 
Gradient Boost (XGBoost), Light GBM,
Random Forest, Support Vector Regression (SVR),
and Long Short-Term Memory (LSTM). We found XGBoost to be the best-
performing individual method, and focused on hyper-parameter tuning
via grid search. We further employed ensemble prediction to further
improve our results, achieving a notable RMSLE score of 0.425 within our
compressed project time-frame.

# Related Work[^2]
[^2]: Hardy's section, and others

The store-sales prediction problem can be directly
formulated as a multivariate
multiple time-series regression problems. Prior to the
modern age of machine learning, Auto-Regressive Moving Average (ARMA)
was one of the most well-known technique, first proposed by
[@whittle1951hypothesis] in his Ph.D thesis, and later popularized by
Box and Jenkins [@RePEc:bla:jtsera:v:37:y:2016:i:5:p:709-711],
according to Wikipedia
[@enwiki:1108230487].
ARMA provides a succinct
description of a (weakly) stationary stochastic process in terms of
two variables, one for the auto-regression (AR), and the second for
the moving average (MA). ARIMA, where "I" stands for Integrated and
"S" stands for seasonal, are variants of ARMA
appropriate for cases when data show evidence of non-stationarity, such
as a long-term upward trend, and seasonality.

Random forest, first proposed by
Ho [@598994] in 2008, is a well-established technique that relies on
an ensemble learning method for classification or regression
based on a collection of decision trees at training time.
For regression tasks, usually the average prediction of the individual trees
is taken, thereby correcting the tendency of individual trees to
overfit to their training set.

XGBoost 
[@Chen:2016:XST:2939672.2939785], first released in 2016,
is a popular open-source software library which provides a gradient
boosting decision tree (GBDT) framework for a plethora of program languages.

LightGBM [@NIPS2017_6449f44a]
is a gradient boosting decision tree (GBDT) developed by Microsoft with
the explicit goal of speeding up the training process of conventional 
GBDT by up to 20X while achieving almost the same accuracy. The huge
runtime efficiency makes it a popular GBDT technique in recent years.

- Support Vector Machine(SVM) /Support Vector Regression (SVR)
- Long Short-Term Memory (LSTM)

# Data Preparation[^3]
[^3]:Hardy's section

Our main dataset contained the daily sales figures of 54 stores of
Corporación Favorita,
from January 1st, 2013 to August 15th, 2017, except Christmas Days when
the stores were all closed. We are also given the locations (city and state)
of each store.  There are a total of 33 different
product families, from \texttt{Automobile} to \texttt{Seafood}. Note that
not all stores sell all products. Moreover, there were some stores that
opened only after the data collection has begun, and sometimes the sales
figures of certain stores were missing over a few months. All stores in the
dataset were still in business as of August 15th, 2017,
After properly dealing with missing data, Christmas, and
stores that were yet to open, we would have a dataset made of
$(\text{\#days}) \times (\text{\#stores}) \times (\text{\#families)} = 1688 \times 54 \times 33 = 3008016$ numbers.
We were also given the daily oil price over the same period of time, as well
as the dates of the regional or national holidays. These information  could
have a tremendous impact on the accuracy of our prediction, and hence must be
considered.

Our job is to predict the sales figures for each store between August 16th,
2017 and August 31st, 2017, inclusive.

The preprocessing of data will include checking for missing values and
if found, imputing them so that no data is disregarded.
We have also checked for frequency distribution of data elements as part
of Data exploration. We have calculated and plotted correlation mapping to
establish the correlations between different input and output parameters
such as holiday events, oil prices, transactions per store type,
dates of salary, natural calamities etc.

# Experimental Setup[^4]
[^4]: Hardy's section

Talk about the experimental setup, including how to 

## Part I[^5]
[^5]: Loukya's section

- Linear regression
- XGBoost
- LightGBM
- Grid search technique to improve XGBoost performance
- CatBoost

## Part II[^6]
[^6]: Cody's section

- LSTM, architecture, slightly different flow

## Part III[^7]
[^7]: Suhas's section

- Ensemble approach. You can quote this [@rokach2010ensemble]
- Final result
- Visualization

# Discussion[^8]
[^8]: TBD

- Possible enhancements

# Conclusions[^9]
[^9]: Hardy

In this work, we have presented LiBerTY, an ensemble regression engine
that successfully predicts the store sales, which successfully
employed a variety of data
engineering and machine learning techniques.

# References
