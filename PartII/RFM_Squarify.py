#!/usr/bin/env python
# coding: utf-8

# # Squarify plot for complete efood clientele

# In[2]:


import pandas as pd
import numpy as np
from scipy import stats
from datetime import timedelta
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import squarify


# In[3]:


# Load the RFM_Segment saved from "efood_PartII"
df=pd.read_csv('RFM_Segment.csv')
df = df.reset_index(drop=True)
df = df.drop("Unnamed: 0", axis = 1 )
df.head()


# # Calculate RFM Score

# In[4]:


# count num of unique segments
df_count_unique = df.groupby('RFMClass')['RFMClass'].nunique()
print(df_count_unique.sum())
# Calculate RFM Score
df['RFM_Score'] = df[['R_Quartile', 'F_Quartile', 'M_Quartile']].sum(axis=1)
print(df['RFM_Score'].head())


# In[5]:


# Define df_level function
#def df_level(df):
 #   if df['RFM_Score'] >= 10:
  #      return 'Best Customers'
   # elif((df['F_Quartile'] >= 3) & (df['R_Quartile'] >= 2)):
    #    return 'Loyal Customers'
   # elif ((df['R_Quartile'] <= 2 )):
    #    return ' Customers at the verge of Churning'
   # elif ((df['R_Quartile'] <= 2) & (df['F_Quartile'] >= 1) & (df['M_Quartile'] <= 3)):
    #    return ' Refined Customers at the verge of Churning'
   # else:
    #    return ' Lost Customers'
# Create a new varieable df_level
#df['RFM_level'] = df.apply(df_level, axis = 1)
# print the top 15 rows
#df.head(10)


# In[6]:


def df_level(df):
    if df['RFM_Score'] >= 9:
        return 'Best Customers'
    elif ((df['RFM_Score'] >= 8) and (df['RFM_Score'] < 9)):
        return 'Champions'
    elif ((df['RFM_Score'] >= 7) and (df['RFM_Score'] < 8)):
        return 'Loyal Customers'
    elif ((df['RFM_Score'] >= 6) and (df['RFM_Score'] < 7)):
        return 'Potential'
    elif ((df['RFM_Score'] >= 5) and (df['RFM_Score'] < 6)):
        return 'Promising'
    elif ((df['RFM_Score'] >= 4) and (df['RFM_Score'] < 5)):
        return 'Requires Attention'
    else:
        return 'Lost Customers'
# create new variable df_level
df['RFM_level'] = df.apply(df_level, axis = 1)
# print the top 15 rows
df.head(10)


# In[7]:


# Calculate the average values of each RFM_level and return a size for each segment
df_level_agg = df.groupby('RFM_level').agg({
    'Recency' : 'mean',
    'frequency': 'mean',
    'avg_order_value': ['mean', 'count']
}).round(1)
# Print the aggregated dataset
print(df_level_agg)


# # Plot the Squarify plot

# In[16]:


df_level_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
# Define the colour palette
colors = ['#377EB8', '#1F78B4', '#A6CEE3', '#FFC300', '#FF69B4', '#FF5733', '#8B0000']
# Create the plot and resize it 
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(16, 9)
squarify.plot(sizes=df_level_agg['Count'],
              label=['Best Customers',
                     'Champions',
                     'Loyal Customers',
                     'Potential',
                     'Promising',
                     'Requires Attention',
                     'Lost Customers'],alpha=.6, color=colors,)
plt.title("RFM Segments", fontsize=18,fontweight="bold")
plt.axis('off')
plt.show()


# # Squarify plot for efood clientele that ordered breakfast

# In[17]:


df2 = pd.read_csv('RFM_Breakfast.csv')
df2.drop("Unnamed: 0", axis = 1 )


# In[18]:


# Filter the dataset for breakfast cuisine
RFM_Breakfast = df2[df2['cuisine'] == 'Breakfast'].copy()

# Reset the index of the new dataset
RFM_Breakfast.reset_index(drop=True, inplace=True)
RFM_Breakfast = RFM_Breakfast.drop(columns=['Unnamed: 0'])
RFM_Breakfast.head()


# # Calculate RFM Score

# In[19]:


# count num of unique segments
RFM_Breakfast_count_unique = RFM_Breakfast.groupby('RFMClass')['RFMClass'].nunique()
print(RFM_Breakfast_count_unique.sum())
# Calculate RFM Score
RFM_Breakfast['RFM_Score'] = RFM_Breakfast[['R_Quartile', 'F_Quartile', 'M_Quartile']].sum(axis=1)
print(RFM_Breakfast['RFM_Score'].head())


# In[20]:


def RFM_Breakfast_level(RFM_Breakfast):
    if  RFM_Breakfast['RFM_Score'] >= 9:
        return 'Best Customers'
    elif  ((RFM_Breakfast['RFM_Score'] >= 8) and (RFM_Breakfast['RFM_Score'] < 9)):
        return 'Champions'
    elif ((RFM_Breakfast['RFM_Score'] >= 7) and (RFM_Breakfast['RFM_Score'] < 8)):
        return 'Loyal Customers'
    elif ((RFM_Breakfast['RFM_Score'] >= 6) and (RFM_Breakfast['RFM_Score'] < 7)):
        return 'Potential'
    elif ((RFM_Breakfast['RFM_Score'] >= 5) and (RFM_Breakfast['RFM_Score'] < 6)):
        return 'Promising'
    elif ((RFM_Breakfast['RFM_Score'] >= 4) and (RFM_Breakfast['RFM_Score'] < 5)):
        return 'Requires Attention'
    else :
        return 'Lost Customers'
RFM_Breakfast['RFM_Breakfast_level'] = RFM_Breakfast.apply(RFM_Breakfast_level, axis=1)
print(RFM_Breakfast.head(10))


# In[21]:


# Calculate the average values of each RFM_level and return a size for each segment
RFM_Breakfast_level_agg = RFM_Breakfast.groupby('RFM_Breakfast_level').agg({
    'Recency' : 'mean',
    'frequency': 'mean',
    'avg_order_value': ['mean', 'count']
}).round(1)
# Print the aggregated dataset
print(RFM_Breakfast_level_agg)


# Squarify for Breakfast customers

# In[24]:


RFM_Breakfast_level_agg.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
# Define the colour palette
colors = ['#377EB8', '#1F78B4', '#A6CEE3', '#FFC300', '#FF69B4', '#FF5733', '#8B0000']
# Create the plot and resize it 
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(16, 9)
squarify.plot(sizes=RFM_Breakfast_level_agg['Count'],
              label=['Best Customers',
                     'Champions',
                     'Loyal Customers',
                     'Potential',
                     'Promising',
                     'Requires Attention',
                     'Lost Customers'],alpha=.6, color=colors,)
plt.title("RFM Segments Breakfast Customers", fontsize=18,fontweight="bold")
plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




