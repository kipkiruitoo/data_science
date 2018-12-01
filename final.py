
# coding: utf-8

# In[1]:


# importing pandas
import pandas as pd
# importing numpy
import numpy as np

# import matplotlib
import matplotlib.pyplot as plt


# In[2]:


# import data in the csv file

data = pd.read_csv("C:/Users/davis/Downloads/California_cities.csv")


# In[3]:


# Another way of checking for missing values
data.isnull().sum()


# In[4]:


# Replace all missing values with the median of each column
# elevation_m column
data['elevation_m'].fillna(data['elevation_m'].median(), inplace=True)
# elevation_ft column
data['elevation_ft'].fillna(data['elevation_ft'].median(), inplace=True)
# area_total_sq_mi column
data['area_total_sq_mi'].fillna(data['area_total_sq_mi'].median(), inplace=True)
# area_water_sq_mi
data['area_water_sq_mi'].fillna(data['area_water_sq_mi'].median(), inplace=True)
# area_total_km2
data['area_total_km2'].fillna(data['area_total_km2'].median(), inplace=True)
# area_land_km2
data['area_land_km2'].fillna(data['area_land_km2'].median(), inplace=True)
# area_water_km2
data['area_water_km2'].fillna(data['area_water_km2'].median(), inplace=True)
# area_water_percent
data['area_water_percent'].fillna(data['area_water_percent'].median(), inplace=True)


# In[5]:


# check if there are any remaining missing values
data.isnull().sum()


# In[6]:


# mean, max, min, std, medium of the ten lowest values of 
# latitude values
data['latd'].nsmallest(10).describe()


# In[7]:


#Display the ten cities with the least value of elevation in Meterslowest
lowest_cities = data['elevation_m'].nsmallest(10)
data.loc[data['elevation_m'].isin(lowest_cities), 'city']


# In[8]:


# get top ten cities witth highest population totals

highest = data['population_total'].nlargest(10)

# gets rows with the ten highest populations
data.loc[data['population_total'].isin(highest), 'city']


# In[9]:


# Plot the relationship of the top ten highest areas in feet with their respective population totals

highest_areas = data['area_total_km2'].nlargest(10)
# gets rows with the ten highest total areas 
p = data.loc[data['area_total_km2'].isin(highest_areas)]

p


# In[26]:


#Explore the relationship between Area in km2 and population_total
p.plot( x='area_total_km2', y='population_total',style='*')  
plt.title('area in km2 and population_total')  
plt.xlabel('Area in km2')  
plt.ylabel('Total Population')  
plt.show() 


# In[13]:


# plot the relationship of top ten highest in elavation_ft and 

highest_elevation = data['elevation_ft'].nlargest(10)

a = data.loc[data['elevation_ft'].isin(highest_elevation)]

a


# In[24]:


#Explore the relationship between elevation_ft and population_total
a.plot( x='elevation_ft', y='population_total',style='*')  
plt.title('elevation_ft and population_total')  
plt.xlabel('Elavation in ft')  
plt.ylabel('Total Population')  
plt.show() 


# In[50]:


# 8.	Plot a histogram of the area_total_sq_mi and discribe 
# the patter of the data (you can use 20 bins for the histogram plot)
areas = data['area_total_sq_mi']


# In[51]:


areas.plot.hist(bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Area square miles histogram')
plt.xlabel('Number of Areas')
plt.ylabel('Area in square miles')
plt.grid(axis='y', alpha=0.75)


# In[63]:


# 9.Normalize the values of the area_total_sq_mi, 
# plot a histogram of the values and describe the pattern of the 
# data after normalization.
normalized = data['area_total_sq_mi'].value_counts(normalize=True)

normalized.plot.hist(bins=20, rwidth=1.0,
                   color='#607c8e')
plt.title('Normalized')
plt.xlabel('Number of Areas')
plt.ylabel('Area in square miles')
plt.grid(axis='y', alpha=0.75)


# In[62]:


# 10.	Using a plot analyze the relationship of the population total by the area 
# total in sq mi
data.plot( x='area_total_sq_mi', y='population_total',style='*')  
plt.title('elevation_ft and population_total')  
plt.xlabel('area total in sq mi')  
plt.ylabel('Total Population')  
plt.show() 


# In[65]:


import statsmodels.api as sm


# In[67]:


X = data['area_total_sq_mi']
Y = data['population_total']

model = sm.OLS(Y, X).fit()

predictions = model.predict(X)

model.summary()


# In[73]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

Areas = data['area_total_sq_mi'].values
Population = data['population_total'].values



# X and Y Values
X = np.array([Areas]).T
Y = np.array(Population)

# Model Intialization
reg = LinearRegression()
# Data Fitting
reg = reg.fit(X, Y)
# Y Prediction
Y_pred = reg.predict(X)

# Model Evaluation
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
r2 = reg.score(X, Y)

print(rmse)
print(r2)

