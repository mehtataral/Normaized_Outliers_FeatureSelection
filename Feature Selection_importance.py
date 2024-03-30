# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:58:30 2023

@author: Livewire
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# https://www.analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/

df = pd.read_csv(r"E:\Taral\DATA_SET\DATA_SET\classification data\Churn_Modelling.csv")
df

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

x.drop(columns = ["RowNumber","Surname","CustomerId"],inplace =True)
x.info()

from sklearn.preprocessing import LabelEncoder
lr = LabelEncoder()
x.Gender = lr.fit_transform(x.Gender)
x.Geography = lr.fit_transform(x.Geography)


x.Age.fillna(x.Age.mean(),inplace =True)

x.isnull().sum()

#*******************Information Gain*******************************
# Information Gain is a concept commonly used in machine learning and 
# statistics for feature selection.

# It helps quantify the amount of information a particular feature provides about
# the class labels of a dataset.

#It's particularly useful in decision tree algorithms and 
# other approaches that involve recursive partitioning of data.

# When working with datasets that contain multiple features, 
# not all features contribute equally to the predictive power of a model. 
# Some features might be redundant or irrelevant,
# and using them could even introduce noise to the model.

# Information Gain can be used to rank features based on their ability to 
# differentiate between different classes. Features with higher Information Gain 
# are considered more useful for classification because they help to distinguish 
# between different class labels.


from sklearn.feature_selection import mutual_info_classif
importance = mutual_info_classif(x, y)
importance


feature_importance = pd.Series(importance,x.columns[0:len(x.columns)])
feature_importance

feature_importance.plot(kind ="barh",color ="teal")


#*******************Chi-square Test***************************
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

x.info()

chi2_feature = SelectKBest(chi2,k=3)
chi2_feature

x_kbest_feature =chi2_feature.fit_transform(x,y)

x.shape[1]
x_kbest_feature.shape[1]

#*******************Fisher score***************************
from skfeature.function.similarity_based import fisher_score

import numpy as np
from scipy.stats import skew, kurtosis

# Generate random data (example data)
data = np.random.normal(loc=0, scale=1, size=1000)

# Calculate skewness and kurtosis
skewness = skew(data)
# Skewness = (mean - median) / sd
# A skewness coefficient of 0 indicates that the distribution is symmetric. 
# A positive skewness coefficient indicates that the distribution is positively skewed, 
# while a negative skewness coefficient indicates that the distribution is negatively skewed.
kurt = kurtosis(data)
# The normal distribution has a kurtosis of 3
# A distribution with a kurtosis greater than 3 is said to be leptokurtic, which means that it has a sharper peak and fatter tails than the normal distribution. 
# A distribution with a kurtosis less than 3 is said to be platykurtic, which means that it has a flatter peak and thinner tails than the normal distribution.
# Kurtosis = (moment 4) / (moment 2)^2

print("Skewness:", skewness)
print("Kurtosis:", kurt)
# ---------------------------- VarianceThreshold ---------------------------------
from sklearn.feature_selection import VarianceThreshold
# Those features which contain constant values (i.e. only one value for all the outputs or target values) in the dataset are known as Constant Features. These features don’t provide any information to the target feature.

# Variance Threshold is a feature selector that removes all the low variance features from the dataset that are of no great use in modeling.

# High Variance in predictors: Good for the model
# Low Variance predictors: Not good for model
# Default Value of Threshold is 0
# If Variance Threshold = 0 (Remove Constant Features )

# If Variance Threshold > 0 (Remove Quasi-Constant Features )
# Assuming X is your feature matrix
# Create a VarianceThreshold object with a threshold (e.g., 0.0 for features with zero variance)
selector = VarianceThreshold(threshold=0.0)

# Fit the selector to your data
selector.fit(X)

# Get the indices of non-constant features
non_constant_indices = selector.get_support(indices=True)

# Filter your feature matrix to keep only non-constant features
X_filtered = selector.transform(X)

# If you need to get the names of the non-constant features (assuming X is a DataFrame)
non_constant_feature_names = X.columns[non_constant_indices]

# Now, X_filtered contains only the non-constant features

