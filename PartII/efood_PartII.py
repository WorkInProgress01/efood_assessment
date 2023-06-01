#!/usr/bin/env python
# coding: utf-8

# # RFM Analysis of efood customers for January 2022

# In[40]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
customer_data = pd.read_csv('Assessment.csv')

# Calculate the frequency of orders for each customer
order_frequency = customer_data.groupby('user_id')['order_id'].count().reset_index()
order_frequency.columns = ['user_id', 'frequency']


# Calculate the average order value for each customer
average_order_value = customer_data.groupby('user_id')['amount'].mean().reset_index()
average_order_value.columns = ['user_id', 'avg_order_value']

# Convert order_timestamp to datetime format
customer_data['order_timestamp'] = pd.to_datetime(customer_data['order_timestamp'])
latest_date = customer_data['order_timestamp'].max()

# Calculate the recency for each customer
customer_data['Last_Purchase_Date'] = customer_data.groupby('user_id')['order_timestamp'].transform('max')
customer_data['Recency'] = (latest_date - customer_data['Last_Purchase_Date']).dt.days

# Merge the frequency, average order value, and recency dataframes
customer_segments = pd.merge(order_frequency, average_order_value, on='user_id')
customer_segments = pd.merge(customer_segments, customer_data[['user_id', 'Recency']], on='user_id')

# Define the segments based on frequency, average order value, and recency
breakpoint_frequency = customer_segments['frequency'].quantile(0.70)
breakpoint_order_value = customer_segments['avg_order_value'].quantile(0.30)
breakpoint_recency = customer_segments['Recency'].quantile(0.50)



# In[41]:


# Merge the frequency, average order value, and recency dataframes
RFM_table = pd.merge(order_frequency, average_order_value, on='user_id')
RFM_table = pd.merge(RFM_table, customer_data[['user_id', 'Recency']], on='user_id')

RFM_table.head()


# # Plot the distributions of 'frequency', 'avg_order_value', 'Recency'

# In[44]:


# Calculate the average values
avg_recency = RFM_table['Recency'].mean()
avg_frequency = RFM_table['frequency'].mean()
avg_avg_order_value = RFM_table['avg_order_value'].mean()

# Print the average values
print("Average Recency:", avg_recency)
print("Average Frequency:", avg_frequency)
print("Average Average Order Value:", avg_avg_order_value)


# In[45]:


# Plotting RFM distributions
plt.figure(figsize=(12,10))
# Plot distribution of R
plt.subplot(3,1,1); sns.distplot(RFM_table['Recency'])
# Plot distribution of F
plt.subplot(3,1,2); sns.distplot(RFM_table['frequency'])
# Plot distribution of M (avg_order_value)
plt.subplot(3,1,3); sns.displot(RFM_table['avg_order_value'])
plt.show()


# In[46]:


# Aggregate the metrics at the user_id level
RFM_table = customer_segments.groupby('user_id').agg({
    'frequency': 'sum',
    'avg_order_value': 'mean',
    'Recency': 'max'
}).reset_index()


# In[47]:


# Calculate the quantiles for RFM segmentaion
quantiles = RFM_table.quantile(q=[0.25,0.5,0.75])
quantiles


# In[48]:


quantiles = quantiles.to_dict()


# In[49]:


# RFM segmentation
RFM_Segment = RFM_table.copy()


# In[50]:


# Function for assigning R class based on quartiles
def R_Class(x, p, d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1


# In[51]:


# Function for assigning F/M class based on quartiles
def FM_Class(x, p, d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4


# In[52]:


# Assign R, F, and M quartiles
RFM_Segment['R_Quartile'] = RFM_Segment['Recency'].apply(R_Class, args=('Recency', quantiles,))
RFM_Segment['F_Quartile'] = RFM_Segment['frequency'].apply(FM_Class, args=('frequency', quantiles,))
RFM_Segment['M_Quartile'] = RFM_Segment['avg_order_value'].apply(FM_Class, args=('avg_order_value', quantiles,))


# In[53]:


# Create RFMClass column
RFM_Segment['RFMClass'] = RFM_Segment['R_Quartile'].map(str) + RFM_Segment['F_Quartile'].map(str) + RFM_Segment['M_Quartile'].map(str)


# In[54]:


# Save table to use for Squarify plot.
# RFM_Segment.to_csv('RFM_Segment.csv')


# In[55]:


RFM_Segment = pd.merge(RFM_Segment, customer_data[['user_id', 'cuisine']], on='user_id')


# In[56]:


# Print the RFM table
RFM_Segment.head()


# In[57]:


# Save the table to use for Squarify plot for breakfast cuisine
RFM_Segment.to_csv('RFM_Breakfast.csv')


# # Which are our best customers ?

# In[58]:


# RFMClass = 444 , best customers by Rank
RFM_Segment[RFM_Segment['RFMClass']=='444'].head(20)


# # Which customers are at the verge of churning ?

# In[59]:


# Customers who's recency value is high 
RFM_Segment[RFM_Segment['R_Quartile'] <= 2 ].sort_values('avg_order_value', ascending=False).head(5)


# # Refined Customers at the verge of churning.

# In[60]:


# customers that haven't ordered recently but are within the price range of breakfast.
RFM_Segment[(RFM_Segment['R_Quartile'] <= 2) & (RFM_Segment['F_Quartile'] >= 1) & (RFM_Segment['M_Quartile'] <= 3) ].sort_values('avg_order_value', ascending=False).head(5)


# # "Lost" Customers

# In[61]:


# RFMClass = 111
RFM_Segment[RFM_Segment['RFMClass']=='111'].sort_values('Recency',ascending=False).head(5)


# # Loyal Customers

# In[62]:


#Customers with high frequency, recency

RFM_Segment[(RFM_Segment['F_Quartile'] >= 3) & (RFM_Segment['R_Quartile'] >= 2)].head(5)


# # Loyal Customers in terms of Breakfast

# In[63]:


RFM_Segment[(RFM_Segment['F_Quartile'] >= 3) & (RFM_Segment['R_Quartile'] >= 2) & (RFM_Segment['cuisine']=='Breakfast')].head(5)


# In[64]:


# Filter the RFM_Segment table for loyal customers with high frequency and recency, considering the 'Breakfast' cuisine
target_segment = RFM_Segment[(RFM_Segment['F_Quartile'] >= 3) & (RFM_Segment['R_Quartile'] >= 2) & (RFM_Segment['cuisine'] == 'Breakfast')]

# Print the target segment for the marketing campaign
print("Target Segment for 'Breakfast' cuisine marketing campaign:")
target_segment


# In[65]:


RFM_Segment.shape


# # Mapping dictionary for RFM Classes.

# In[66]:


# Create a mapping dictionary for RFM classes
rfm_labels = {
    '444': 'Best Customers',
    '111': 'Lost Customers',
    '113': 'HR, LF, LM',
    '114': 'HR, LF, HM',
    '123': 'HR, HF, LM',
    '124': 'HR, HF, HM',
    '133': 'HR, HF, LM',
    '134': 'HR, HF, HM',
    '143': 'HR, HF, LM',
    '144': 'HR, HF, HM',
    '213': 'LR, LF, LM',
    '214': 'LR, LF, HM',
    '223': 'LR, LF, LM',
    '224': 'LR, LF, HM',
    '233': 'LR, LF, LM',
    '234': 'LR, LF, HM',
    '243': 'LR, LF, LM',
    '244': 'LR, LF, HM',
    '313': 'LR, HF, LM',
    '314': 'LR, HF, HM',
    '323': 'LR, HF, LM',
    '324': 'LR, HF, HM',
    '333': 'LR, HF, LM',
    '334': 'LR, HF, HM',
    '343': 'LR, HF, LM',
    '344': 'LR, HF, HM',
    '413': 'LR, LF, LM',
    '414': 'LR, LF, HM',
    '423': 'LR, HF, LM',
    '424': 'LR, HF, HM',
    '433': 'LR, LF, LM',
    '434': 'LR, LF, HM',
    '443': 'LR, HF, LM',
    
}

# Map RFM classes to labels
RFM_Segment['RFMLabel'] = RFM_Segment['RFMClass'].map(rfm_labels)


# # Plots

# ## Bar Plot of RFM segments

# In[67]:


# Use for custom plots 
#rfm_order=['Best Customers','Lost Customers', 'HR, LF, LM', 'HR, LF, HM', 'HR, HF, LM', 'HR, HF, HM', 'LR, LF, LM', 'LR, LF, HM', 'LR, HF, LM', 
           #'LR, HF, HM', 'LR, LF, LM', 'LR, LF, HM', 'LR, HF, LM', 'LR, HF, HM']


# In[68]:


# Plot the counts of each RFM segment
plt.figure(figsize=(15, 8))
sns.countplot(x='RFMLabel', data=RFM_Segment, palette='pink')
plt.xlabel('RFM Segment')
plt.ylabel('Count')
plt.title('Distribution of RFM Segments')
plt.xticks(rotation=0, ha = 'right')
plt.ylim(0,30000)
plt.tight_layout()
plt.show()


# ## Scatter plot of Recency vs. Frequency

# In[69]:


plt.figure(figsize=(10, 10))
sns.scatterplot(x='Recency', y='frequency', data=RFM_Segment, color='red')
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.title('Recency vs. Frequency')
plt.show()


# ## Box plot of Monetary value:

# In[70]:


plt.figure(figsize=(8, 6))
sns.violinplot(y='avg_order_value', data=RFM_Segment, color='pink')
plt.ylabel('Average Order Value')
plt.title('Distribution of Average Order Value')
plt.show()


# ## Bar plot of cuisine distribution within RFM segments:

# In[71]:


plt.figure(figsize=(20, 6))
sns.countplot(x='RFMLabel', hue='cuisine', data=RFM_Segment, palette='husl')
plt.xlabel('RFM Class')
plt.ylabel('Count')
plt.title('Cuisine Distribution within RFM Segments')
plt.xticks(rotation=0, ha = 'right')
plt.legend(title='Cuisine')
plt.show()


# 

# In[ ]:





# In[ ]:




