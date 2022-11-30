
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
given. It employed a variety of data engineering and machine learning
techniques to most 

# Introduction[^1]
[^1]: Suhas's section

The ability to predict the sales of a variety of stores is highly sought
out in supply chain logistics, as it finds applications in increasing
customer satisfaction and reducing food waste.  We are proposing use of
multiple Supervised Learning methods to predict the sales of stores based on
time series dataset of Corporación Favorita, an Ecuador based grocery retailer. Ecuador is a country whose economy is strongly dependent on the oil and
fluctuates with the price of oil.

We are planning to use dataset from an ongoing Kaggle competition [@kaggle],
``Store Sales -- Time Series Forecasting''. The dataset includes multiple
csv sheets of time series data. We will try to evaluate the different
aspects that might impact the sales in a store like Holiday seasons,
Oil prices and historical sales data from a variety of stores.
The preprocessing of data will include checking for missing values and
if found, imputing them so that no data is disregarded.
We have also checked for frequency distribution of data elements as part
of Data exploration. We have calculated and plotted correlation mapping to
establish the correlations between different input and output parameters
such as holiday events, oil prices, transactions per store type,
dates of salary, natural calamities etc.

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

Cite reference to these models.

XGBoost 
[@Chen:2016:XST:2939672.2939785]
is a popular open-source software library which provides a gradient
boosting framework for C++, Java, Python, R, Julia, Perl, and Scala.

LightGBM [@NIPS2017_6449f44a]
is a gradient boosting decision tree (GBDT) implemented by Microsoft which
focused on speeding up the training process of conventional 
GBDT by up to 20X while achieving almost the same accuracy. The huge
runtime efficiency makes it a popular GBDT technique in recent years.

- Linear Regression
- Gradient Boost(XGBoost)
- Random Forest
- Support Vector Machine(SVM) /Support Vector Regression (SVR)
- Long Short-Term Memory (LSTM)

# Data Preparation[^3]

[^3]:Hardy's section

# Experimental Setup[^4]

[^4]: Hardy's section

Talk about the experimental setup, including how to 

## Part I[^5]
[^5]: Loukya's section

- Linear regression
- XGBoost
- LightGBM
- Grid search technique to improve XGBoost performance

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
