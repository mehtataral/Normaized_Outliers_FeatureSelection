# 1. Take DataSet

# 2. Check EDA PART , Which includes
# 	cleaning part,
# 	check each columns whether there is skewness present or not
# 	removing outliers

# 3.Analyze whole data set (Pandas ,Matplotlib,Seaborn..etc)

# 4.Prediction Part(ALgorithm)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_excel(r"C:\Users\user\Downloads\archive\superstore.xls")
print(df)
df.info()
df.isnull().sum()

df.drop("Row ID",axis ="columns",inplace =True)
df
df.drop("Order ID",axis ="columns",inplace =True)
df


#how to create new columns
df.loc[(df.Profit <0),"Result_profit"] ="Loss"
print(df)
df.loc[(df.Profit == 0),"Result_profit"] ="No-Profit\Loss"
print(df)
df.loc[(df.Profit >0),"Result_profit"] ="Profit"
print(df)
#******************************************************************************
df.Country.unique()

df["Ship Mode"].unique()

df.State.unique()
df.State.nunique()
#******************************************************************************
x = df.loc[(df.State =="New York")]
x

x = df.loc[(df.State =="New York")&(df["Sub-Category"]=="Phones")]
x.Profit.max()
x.Profit.min()
x.Profit.sort_values()

x.Profit.nlargest(5)
x.Profit.nsmallest(2)
x["Year"] =x["Order Date"].dt.year.astype(str)

sns.scatterplot(data =x,x=x["Year"],y= x.Profit,hue =x.Category)
sns.barplot(data =x,x=x["Year"],y= x.Profit,hue =x.Category)
sns.lineplot(data =x,x=x["Year"],y= x.Profit,hue =x.Category)
#******************************************************************************
x.City.nunique()

y = x.loc[(x.State =="New York") & (x["Profit"]>0) & 
           (x["Category"] =="Technology")]


y["Sub-Category"].nunique()

y.loc[(y["Sub-Category"]=="Copiers")] #here we are getting data of copiers

ax =sns.barplot(data= y,x = y["Sub-Category"],y=y.Profit)  
ax.bar_label(ax.containers[0])
#in bar graph middle line indicating avg of data
#here we are getting avg of profit - 174+3919+59+329+249+1014+89 = 5833. 5833/7 =833.28

ax =sns.barplot(data= y,x = y["Sub-Category"],y=y.Profit,hue =y["Year"])
ax.bar_label(ax.containers[0])

#here we are getting profit of individual year

for data  in y["Year"].unique():
    for da in y["Sub-Category"].unique():
        temp = y.loc[(y["Year"]==data)&(y["Sub-Category"]==da) ]
        print("Year is {} and Sub-Category is {} ".format(data, da),temp.Profit.max())
        # print("Year is {} ".format(data), temp.Profit.min())


#******************************************************************************
a =y.groupby("Sub-Category")
a.first()
a.last()

a = df.groupby("Sub-Category")
a.first()
a.last()

a = y.groupby(["Sub-Category","Year"])
a.first()
a.last()

a = y.groupby(["Year"])
b =a.first()
b =a.last()

y.State
plt.pie(b.Profit,labels =y.Year.unique())
plt.title("Profit in New York")
plt.legend()


df_copy =df.copy()
df_copy
df_copy["Year"] =df_copy["Order Date"].dt.year.astype(str)
df_copy

aa= df_copy.groupby(["State","Year"])
c =aa.first()
c
# for x in df_copy.State.unique():
#     print(x)
#     temp = df_copy.loc[(df_copy.State == x)&(df_copy.Profit >0)]
#     print(temp)
#     plt.pie(temp.Profit,)
#     plt.title("Profit in {}".format(x))
#     plt.legend()
    

state_list =[]
for xi,yi in enumerate(c.index):
    print(xi,yi)
    temp = df_copy.loc[(df_copy.State == yi[0]) & (df_copy.Profit >0)]
    state_list.append(temp)
    # plt.pie(temp.Profit,labels=temp.Year.unique())
    # print(temp)
print(state_list)    
print(state_list[0].State)    
print(state_list[0].City)    
print(state_list[0].Year)  

  

#******************************************************************************
sns.scatterplot(data =df_copy,x=df_copy["Profit"],y= df_copy["Sub-Category"],hue = df_copy["Category"])

#******************************************************************************

df_copy.Result_profit.unique()
df_copy.Result_profit.value_counts()
plt.pie(df_copy.Result_profit.value_counts(),labels =df_copy.Result_profit.unique())
plt.title("SHow Profit / Loss")
plt.legend()

sns.countplot(data =df_copy,x=df_copy["Result_profit"])

#******************************************************************************

# Outliers

df_copy.info()

sns.boxplot(df_copy.Sales)

sns.boxplot(df_copy.Quantity)

sns.boxplot(df_copy.Discount)

sns.boxplot(df_copy.Profit)
#*****************************************************************************
# left skewed distribution  --> mode> median> mean
# Righ skewed distribution  --> mode< median< mean

df_copy.skew()
# if values <  - 0.5 then it is  negative
# whereas  value is > 0.5 then it is  positively skewed 
# if value is between  -0.5 to 0.5 we consider it is normal distribution


import statsmodels.api as sm


fig = sm.qqplot(df_copy.Sales, line='45')
fig = sm.qqplot(df_copy.Quantity, line='45')
fig = sm.qqplot(df_copy.Discount, line='45')
fig = sm.qqplot(df_copy.Profit, line='45')

sns.displot(df_copy.Sales)
plt.hist(df_copy.Sales)


sns.displot(df_copy.Quantity)
plt.hist(df_copy.Quantity)


sns.displot(df_copy.Discount)
plt.hist(df_copy.Discount)


sns.displot(df_copy.Profit)
plt.hist(df_copy.Profit)


from scipy.stats import shapiro 
shapiro(df_copy.Profit)
shapiro(df_copy.Discount)
shapiro(df_copy.Quantity)
shapiro(df_copy.Sales)

from scipy.stats import kstest
kstest(df_copy.Profit, 'norm')
kstest(df_copy.Sales, 'norm')
kstest(df_copy.Quantity, 'norm')
kstest(df_copy.Discount, 'norm')



# *****************************************************************************
            # How to Handle Non-Normal Data
# *****************************************************************************
# Log Transformation

data_log = np.log(df_copy.Sales)
plt.hist(data_log, edgecolor='black')


df_copy.Profit.describe()
dummy = df_copy.loc[df_copy.Profit >0]
data_log = np.log(dummy.Profit)
plt.hist(data_log, edgecolor='black')

df_copy.Discount.describe()
dummy = df_copy.loc[df_copy.Discount >0]
data_log = np.log(df_copy.Discount)
plt.hist(data_log, edgecolor='black')

data_log = np.log(df_copy.Quantity)
plt.hist(data_log, edgecolor='black')


# suare root Transformation 


data_log = np.sqrt(df_copy.Sales)
plt.hist(data_log, edgecolor='black')

data_log = np.sqrt(df_copy.Profit)
plt.hist(data_log, edgecolor='black')

data_log = np.sqrt(df_copy.Discount)
plt.hist(data_log, edgecolor='black')

data_log = np.sqrt(df_copy.Quantity)
plt.hist(data_log, edgecolor='black')

