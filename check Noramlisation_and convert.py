# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:11:43 2023

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel(r"D:\Taral\DATA_SET\DATA_SET\Tableau_PowerBI\global_superstore_2016.xlsx")
df    
     
# 1. (Visual Method) Create a histogram.
# If the histogram is roughly “bell-shaped”, 
# then the data is assumed to be normally distributed.

import math
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt
    
plt.hist(df.Profit, edgecolor='black', bins=10)
plt.hist(df.Sales, edgecolor='black', bins=10)
    

# Plot density plots of the normalized data
df.Profit.plot(kind='density')
plt.show()
#******************************************************
# 2. (Visual Method) Create a Q-Q plot.
import math
import numpy as np
from scipy.stats import lognorm
import statsmodels.api as sm
import matplotlib.pyplot as plt

#create Q-Q plot with 45-degree line added to plot
fig = sm.qqplot(df.Sales, line='45')
fig = sm.qqplot(df.Profit, line='45')

#******************************************************
# 3. (Formal Statistical Test) Perform a Shapiro-Wilk Test.
# If the p-value of the test is greater than α = .05, 
# then the data is assumed to be normally distributed.

import math
import numpy as np
from scipy.stats import shapiro 
from scipy.stats import lognorm


#perform Shapiro-Wilk test for normality
shapiro(df.Profit)
shapiro(df.Sales)


#******************************************************

# 4. (Formal Statistical Test) Perform a Kolmogorov-Smirnov Test.
# If the p-value of the test is greater than α = .05, 
# then the data is assumed to be normally distributed.


import math
import numpy as np
from scipy.stats import kstest
from scipy.stats import lognorm
   

#perform Kolmogorov-Smirnov test for normality
kstest(df.Profit, 'norm')
kstest(df.Sales, 'norm')
 
#******************************************************
5. 
import pandas as pd

# Assume 'data_df' is the DataFrame with normalized features

# Calculate the mean and standard deviation of the normalized data
mean_values = df.Profit.mean()
std_values = df.Profit.std()

# Display the mean and standard deviation
print("Mean:")
print(mean_values)
print("Standard Deviation:")
print(std_values)


#****************** Convert into normal distribution***************

# Why Normalize Data?
# There are several reasons why normalization is important:

# Magnitude: 
# Features with larger magnitudes might dominate the learning process 
# or distance calculations in machine learning algorithms. 
# Normalization prevents this dominance by bringing all features to a similar scale.

# Convergence: 
# Gradient-based optimization algorithms (e.g., in neural networks) 
# converge faster when the features are normalized,
# as the optimization process is less likely to get stuck due
# to widely varying feature magnitudes.

# Distance-based algorithms: 
# Clustering algorithms or distance-based methods 
# (e.g., k-nearest neighbors) are sensitive to the scale of features. 
# Normalization ensures that the distances are calculated accurately 
# and fairly.


#1. Log Transformation in Python  
import numpy as np
import matplotlib.pyplot as plt

#make this example reproducible
  
#create beta distributed random variable with 200 values
data=df.Sales
#create log-transformed data
data_log = np.log(df.Sales)

#define grid of plots
fig, axs = plt.subplots(nrows=1, ncols=2)    

#create histograms
axs[0].hist(data, edgecolor='black')
axs[1].hist(data_log, edgecolor='black')

#add title to each histogram
axs[0].set_title('Original Data')
axs[1].set_title('Log-Transformed Data')

#2 suare root Transformation 

#works on percentage 
data_log = np.sqrt(df.Sales)
data_log1 = np.sqrt(data_log)
plt.hist(data_log1, edgecolor='black')

data_log = np.sqrt(df.Profit)
plt.hist(data_log, edgecolor='black')

data_log = np.sqrt(df.Discount)
plt.hist(data_log, edgecolor='black')

data_log = np.sqrt(df.Quantity)
plt.hist(data_log, edgecolor='black')
   

# ----------------------------------------------------------
df
import pandas as pd
# Min-Max Normalization (Feature Scaling): 
# Scales the feature values between a specified range (usually 0 and 1) using the minimum and maximum values of the feature.
# Assume 'data_df' is your DataFrame with numerical features

# Apply Min-Max normalization to a specific column
def min_max_normalization(column):
    return (column - column.min()) / (column.max() - column.min())

# Normalize a specific column using Min-Max normalization
normalized_column = min_max_normalization(df['Profit'])

# Replace the original column with the normalized values
df['Profit_new'] = normalized_column

# Display the DataFrame with the normalized column
print("Data with Min-Max normalized column:")
print(df)

df['Profit_new'].min()
df['Profit_new'].max()
df['Profit']

# ----------------------------------------------------------
import pandas as pd

# Assume 'data_df' is your DataFrame with numerical features

# Apply Z-Score normalization to a specific column
def z_score_normalization(column):
    return (column - column.mean()) / column.std()

# Normalize a specific column using Z-Score normalization
normalized_column = z_score_normalization(df['Profit'])

# Replace the original column with the normalized values
df['Profit_new_method'] = normalized_column

# Display the DataFrame with the normalized column
print("Data with Z-Score normalized column:")
print(df)
df['Profit_new_method'].min()
df['Profit_new_method'].max()
# --------------------------------------------------------------

import pandas as pd

# Create a sample dataset
data = {
    'Age': [25, 30, 35, 40, 45],
    'Income': [50000, 60000, 70000, 80000, 90000],
    'Education Level': [12, 14, 16, 18, 20]
}

# Create a DataFrame
data_df = pd.DataFrame(data)

# Define the columns to normalize
columns_to_normalize = ['Age', 'Income', 'Education Level']

# Apply Min-Max normalization to the specified columns
for column in columns_to_normalize:
    data_df[column] = (data_df[column] - data_df[column].min()) / (data_df[column].max() - data_df[column].min())

# Display the normalized DataFrame
print("Normalized Data:")
print(data_df)
