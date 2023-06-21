#!/usr/bin/env python
# coding: utf-8

# # EDA

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy.stats as ss
from scipy.stats import f_oneway, norm
from collections import Counter
import math
from itertools import product

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, ConfusionMatrixDisplay, recall_score, precision_score

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.one_hot import OneHotEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import statsmodels.api as sm
# from pycaret.classification import *

from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


# In[31]:


df = pd.read_csv('fraud_oracle.csv')
df


# In[32]:


df.head()


# In[33]:


df.info()


# In[34]:


df.isnull().sum()


# In[35]:


for column in df:
    if column == 'PolicyNumber':
        pass
    else:
        print(column)
        print(sorted(df[column].unique()),"\n")
    


# # Issues
# ### DayOfWeekClaimed, MonthClaimed, and Age contains 0
# ### PolicyType is a merge of VehicleCategory and BasePolicy
# ### PolicyNumber is just a row number (similar to customer id)

# In[36]:


df[df['DayOfWeekClaimed']=='0']


# In[37]:


df[df['MonthClaimed']=='0']


# In[38]:


print(df[df['Age']==0].shape)
df[df['Age']==0].sample(3)


# In[39]:


# Drop the PolicyNumber column
df = df.drop(columns='PolicyNumber')


# In[40]:


df


# # Cleansing
# 

# In[41]:


# Drop the DayOfWeekClaimed & MonthClaimed == 0, they are on 
#the same row and only in one row.
df = df[~(df['MonthClaimed']=='0')]


# In[42]:


df[df['Age']==0]['AgeOfPolicyHolder'].unique()

df['Age'] = df['Age'].replace({0:16.5})


# In[43]:


df


# # Further Analysis & Feature Plots
# 

# In[44]:


ax = sns.countplot(df['FraudFound_P'], 
                   order = df["FraudFound_P"].value_counts().index)

for p, label in zip(ax.patches, df["FraudFound_P"].value_counts().values):
    ax.annotate(label, (p.get_x()+0.320, p.get_height()))
    
ax.set_title('Fraud Status\n0:Non Fraud | 1:Fraud')
ax.text(0, 8000, f'{round(14496/len(df),2)*100}%')
ax.text(1, 8000, f'{round(923/len(df),2)*100}%')
plt.show()


# In[45]:


# Subplots for 'Month', 'DayOfWeek', 'Sex', 'MaritalStatus', 'NumberOfCars',
# 'AccidentArea', 'DriverRating', 'AgentType', 'BasePolicy'.


fig, ax = plt.subplots(3,3, figsize=(15,10))
sns.countplot(data=df, x='Month', hue='FraudFound_P', order=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], ax=ax[0][0])
ax[0][0].set_title('Month')

sns.countplot(data=df, x='DayOfWeek', hue='FraudFound_P', order=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'], ax=ax[0][1])
ax[0][1].set_title('Day')

sns.countplot(data=df, x='Sex', hue='FraudFound_P', ax=ax[0][2])
ax[0][2].set_title('Sex')

sns.countplot(data=df, x='MaritalStatus', hue='FraudFound_P', ax=ax[1][0])
ax[1][0].set_title('MaritalStatus')

sns.countplot(data=df, x='NumberOfCars', hue='FraudFound_P', ax=ax[1][1])
ax[1][1].set_title('NumberOfCars')

sns.countplot(data=df, x='AccidentArea', hue='FraudFound_P', ax=ax[1][2])
ax[1][2].set_title('AccidentArea')

sns.countplot(data=df, x='DriverRating', hue='FraudFound_P', ax=ax[2][0])
ax[2][0].set_title('DriverRating')

sns.countplot(data=df, x='AgentType', hue='FraudFound_P', ax=ax[2][1])
ax[2][1].set_title('Agent Type')

sns.countplot(data=df, x='BasePolicy', hue='FraudFound_P', ax=ax[2][2])
ax[2][2].set_title('Base Policy')

plt.tight_layout()


# In[46]:


# plot by FraudFound, looking for any obvious signs of correlation wrt fraud
gpd_val1 = df.groupby('PolicyType').agg({'FraudFound_P' : 'sum'}).reset_index()
gpd_val2 = df.groupby('PolicyType').agg('count').reset_index()

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(22,6))
sns.barplot(x='PolicyType', y='FraudFound_P', data= gpd_val1, ax=ax1)
sns.barplot(x='PolicyType', y='FraudFound_P', data= gpd_val2, ax=ax2)

ax2.set(ylabel='Total counts')


total_list = pd.concat([gpd_val1, gpd_val2['FraudFound_P'].rename('Total Accidents')], axis=1)
total_list['Percentage by PolicyType']= round((total_list['FraudFound_P']/total_list['Total Accidents'])*100,3)
total_list['Percentage by Total'] = round((total_list['FraudFound_P']/sum(total_list['Total Accidents']))*100,3)

ax2.set(ylabel='Total counts')

data = [['Column total'],
        [sum(total_list['FraudFound_P'])],
       [sum(total_list['Total Accidents'])],
       [sum(total_list['Percentage by PolicyType'])],
       [sum(total_list['Percentage by Total'])]]

nr = pd.DataFrame(data)

nr1 = nr.transpose()
nr1.rename(columns={0:'PolicyType',1:'FraudFound_P',2:'Total Accidents',3:'Percentage by PolicyType', 4:'Percentage by Total'}, inplace=True)
pd.concat([total_list,nr1], ignore_index=True)


# In[47]:


gpd_val1 = df.groupby('VehicleCategory').agg({'FraudFound_P' : 'sum'}).reset_index()
gpd_val6 = df.groupby('VehicleCategory').agg('count').reset_index()
gpd_val3=df.groupby('BasePolicy').agg({'FraudFound_P':'sum'}).reset_index()
gpd_val7 = df.groupby('BasePolicy').agg('count').reset_index()

fig, (ax1, ax3) = plt.subplots(1,2,figsize = (15, 5))
sns.barplot(x='VehicleCategory', y='FraudFound_P', data= gpd_val1, ax=ax1)
#sns.barplot(x='VehicleCategory', y='FraudFound_P', data = gpd_val2, ax=ax2)
sns.barplot(x='BasePolicy', y='FraudFound_P', data = gpd_val3, ax=ax3)
#sns.barplot(x='Basepolicy', y='FraudFound_P', data=gpd_val4, ax=ax4)
None


total_list1 = pd.concat([gpd_val1, gpd_val6['FraudFound_P'].rename('Total Accidents')],axis=1)
total_list1['Percentage by VehicleCategory']= round((total_list1['FraudFound_P']/total_list1['Total Accidents'])*100,3)
total_list1['Percentage by Total'] = round((total_list1['FraudFound_P']/sum(total_list1['Total Accidents']))*100,3)

total_list2 = pd.concat([gpd_val3, gpd_val7['FraudFound_P'].rename('Total Accidents')], axis = 1)
total_list2['Percentage by BasePolicy'] = round((total_list2['FraudFound_P'] / total_list2['Total Accidents'])*100,3)
total_list2['Percentage by Total'] = round((total_list2['FraudFound_P']/sum(total_list2['Total Accidents']))*100,3)

data1 = [['Column total'],
        [sum(total_list1['FraudFound_P'])],
         [sum(total_list1['Total Accidents'])],
         [sum(total_list1['Percentage by VehicleCategory'])],
         [sum(total_list1['Percentage by Total'])]]

data2 = [['Column total'],
         [sum(total_list2['FraudFound_P'])],
         [sum(total_list2['Total Accidents'])],
         [sum(total_list2['Percentage by BasePolicy'])],
         [sum(total_list2['Percentage by Total'])]]

nr1 =pd.DataFrame(data1)
nr1 = nr1.transpose()
nr1.rename(columns={0:'VehicleCategory',1:'FraudFound_P',2:'Total Accidents',3:'Percentage by VehicleCategory', 4:'Percentage by Total'}, inplace=True)
tl1=pd.concat([total_list1,nr1],ignore_index=True)

nr2 = pd.DataFrame(data2)
nr2 = nr2.transpose()
nr2.rename(columns={0:'BasePolicy', 1:'FraudFound_P', 2:'Total Accidents', 3:'Percentage by BasePolicy',4:'Percentage by Total'}, inplace=True)
tl2 = pd.concat([total_list2,nr2],ignore_index=True)


#print(tabulate(tl1, headers=tl1.columns))
#print('  ')
#print(tabulate(tl2, headers=tl2.columns))
display(tl1)
display(tl2)


# In[48]:


#plotting by FraudFound, looking to see if there are anything obvious that correlates to fraud
gpd_val1=df.groupby('Make').agg({'FraudFound_P':'sum'}).reset_index()
gpd_val2=df.groupby('Make').agg('count').reset_index()

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(22, 6))
sns.barplot(x='Make', y='FraudFound_P', data = gpd_val1,ax=ax1)
sns.barplot(x='Make', y='FraudFound_P', data = gpd_val2,ax=ax2)

total_list = pd.concat([gpd_val1, gpd_val2['FraudFound_P'].rename('Total Accidents')],axis=1)
total_list['Percentage by Make']= round((total_list['FraudFound_P']/total_list['Total Accidents'])*100,3)
total_list['Percentage by Total'] = round((total_list['FraudFound_P']/sum(total_list['Total Accidents']))*100,3)

ax2.set(ylabel='Total counts')

data = [['Column total'],
        [sum(total_list['FraudFound_P'])], 
        [sum(total_list['Total Accidents'])], 
        [sum(total_list['Percentage by Make'])], 
        [sum(total_list['Percentage by Total'])]]

nr = pd.DataFrame(data)

nr1 = nr.transpose()
nr1.rename(columns={0:'Make',1:'FraudFound_P',2:'Total Accidents',3:'Percentage by Make',4:'Percentage by Total'}, inplace=True)
pd.concat([total_list,nr1],ignore_index=True)


# # Correlation Matrices

# In[49]:


# Correlation Between Categorical & Target
ob=[]
for data in df.columns:
    if data == 'FraudFound_P':
        ob.append(data)
    if df[data].dtype=='object':
        ob.append(data)
        
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

cramers_df = pd.DataFrame(index=ob)

for x in ob:
    a = []
    for y in ob:
        a.append(cramers_v(df[y], df[x]))
    cramers_df[x] = a
    
    
    plt.figure(figsize=(15,10))
sns.heatmap(cramers_df, annot=True, fmt='.2f')
plt.title('Cramers Corr')
plt.show()


# In[50]:


# Correlation Between Continuous & Target
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=True)
plt.show()


# # Preprocessing

# In[51]:


# Drop Policy Type
df = df.drop(columns='PolicyType')


# In[52]:


df


# # Encode Categorical Data

# In[53]:


df.select_dtypes(include=['object']).dtypes


# In[54]:


#df['AccidentArea'] = df['AccidentArea'].replace({'Urban':1, 'Rural':0})
#df['Sex'] = df['Sex'].replace({'Female':1, 'Male':0})
#df['Fault'] = df['Fault'].replace({'Policy Holder':1, 'Third Party':0})
#df['PoliceReportFiled'] = df['PoliceReportFiled'].replace({'Yes':1, 'No':0})
#df['WitnessPresent'] = df['WitnessPresent'].replace({'Yes':1, 'No':0})
#df['AgentType'] = df['AgentType'].replace({'External':1, 'Internal':0})
#df['Month'] = df['Month'].replace({'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12})
#df['DayOfWeek'] = df['DayOfWeek'].replace({'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7})
#df['MonthClaimed'] = df['MonthClaimed'].replace({'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12})
#df['DayOfWeekClaimed'] = df['DayOfWeekClaimed'].replace({'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7})
#df['PastNumberOfClaims'] = df['PastNumberOfClaims'].replace({'none':0 ,'1':1,'2 to 4':2,'more than 4':3})
#df['NumberOfSuppliments'] = df['NumberOfSuppliments'].replace({'none':0,'1 to 2':1,'3 to 5':2,'more than 5':4})
#df['VehiclePrice'] = df['VehiclePrice'].replace({'less than 20000':0,'20000 to 29000':1,'30000 to 39000':2,
#                                                  '40000 to 59000':3,'60000 to 69000':4,'more than 69000':5})
#df['AgeOfVehicle'] = df['AgeOfVehicle'].replace({'3 years':3,'6 years':6,'7 years':7,'more than 7':8,'5 years':5,'new':0,'4 years':4,'2 years':2})


# In[55]:


col_ordering = [
    {'col':'AccidentArea','mapping':{'Urban':1, 'Rural':0}},
    {'col':'Sex','mapping':{'Female':1, 'Male':0}},
    {'col':'Fault','mapping':{'Policy Holder':1, 'Third Party':0}},
    {'col':'PoliceReportFiled','mapping':{'Yes':1, 'No':0}},
    {'col':'WitnessPresent','mapping':{'Yes':1, 'No':0}},
    {'col':'AgentType','mapping':{'External':1, 'Internal':0}},
    {'col':'Month','mapping':{'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}},
    {'col':'DayOfWeek','mapping':{'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}},
    {'col':'DayOfWeekClaimed','mapping':{'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}},
    {'col':'MonthClaimed','mapping':{'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}},
    {'col':'PastNumberOfClaims','mapping':{'none':0 ,'1':1,'2 to 4':2,'more than 4':3 }},
    {'col':'NumberOfSuppliments','mapping':{'none':0,'1 to 2':1,'3 to 5':2,'more than 5':3}}, 
    {'col':'VehiclePrice','mapping':{'less than 20000':0,'20000 to 29000':1,'30000 to 39000':2,
                                     '40000 to 59000':3,'60000 to 69000':4,'more than 69000':5}},
    {'col':'AgeOfVehicle','mapping':{'3 years':3,'6 years':6,'7 years':7,'more than 7':8,'5 years':5,'new':0,'4 years':4,'2 years':2}},
    {'col':'Days_Policy_Accident','mapping':{'more than 30':4,'15 to 30':3,'none':0,'1 to 7':1,'8 to 15':2}},
    {'col':'Days_Policy_Claim','mapping':{'more than 30':4,'15 to 30':3,'none':0,'1 to 7':1,'8 to 15':2}},
    {'col':'AgeOfPolicyHolder','mapping':{'16 to 17':1,'18 to 20':2,'21 to 25':3,'26 to 30':4,'31 to 35':5,'36 to 40':6,
                                          '41 to 50':7,'51 to 65':8,'over 65':9}},
    {'col':'AddressChange_Claim','mapping':{'no change':0,'under 6 months':1,'1 year':2,'2 to 3 years':3,'4 to 8 years':4}},
    {'col':'NumberOfCars','mapping':{'1 vehicle':1,'2 vehicles':2,'3 to 4':3,'5 to 8':4,'more than 8':5}}
]
ord_encoder = OrdinalEncoder(mapping = col_ordering, return_df=True)


# In[56]:


df2 = ord_encoder.fit_transform(df)


# In[57]:


OHE = OneHotEncoder(cols = ['Make', 'MaritalStatus', 'VehicleCategory','BasePolicy'], use_cat_names=True, return_df=True)
df3 = OHE.fit_transform(df2)


# In[58]:


df3.head()


# In[59]:


cat_var_prod = list(product(df3.columns,df3.columns, repeat = 1 ))

## Creating an empty variable and picking only the p value from the output of Chi-Square test
result = []
for i in cat_var_prod:
    if i[0] != i[1]:
        result.append((i[0],i[1],list(ss.chi2_contingency(pd.crosstab(
                                    df3[i[0]], df3[i[1]])))[1]))
        chi_test_output = pd.DataFrame(result, columns = ['var1', 'var2', 'coeff'])


# In[60]:


chi_test_output2 = chi_test_output[chi_test_output['var1']=='FraudFound_P'].sort_values('coeff').reset_index(drop=True)
def rej_acc(x):
    if x > 0.05:
        Ho = 'A_H0'
    else:
        Ho = 'R_H0'
    return Ho

chi_test_output2['result'] = chi_test_output2['coeff'].apply(rej_acc)
chi_test_output2


# # Split Data

# In[61]:


X = df3.drop(columns='FraudFound_P')
y = df3['FraudFound_P']


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48, stratify=y)


# In[63]:


y_train.value_counts()


# #

# #

# # Model Function

# In[64]:


modelname, acc, recall, precision, f1, roc_auc = [],[],[],[],[],[]
model = {'xgboost':XGBClassifier(objective= 'binary:logistic',eval_metric='aucpr'), 
         'LightGBM':LGBMClassifier(is_unbalance=True),
         'dt': DecisionTreeClassifier(),
         'rf':RandomForestClassifier(),
         'blf':BalancedRandomForestClassifier(),
                   
        }

def fit_model(x_train, x_test, y_train, y_test, sampling):
    for key,value in zip(model, model.values()):
        print(f"Model {key} {sampling}")
        ml_model = value
        ml_model.fit(x_train, y_train)
        y_pred = ml_model.predict(x_test)
        
        modelname.append(f'{key} {sampling}')
        acc.append(accuracy_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred))
        roc_auc.append(roc_auc_score(y_test, y_pred))

def sampling(x_train, y_train, kind='over', ss=0.5):
    if kind == 'over':
        oversample = SMOTE(sampling_strategy=ss)
        X, y = oversample.fit_resample(x_train, y_train)
    elif kind == 'under':
        join_train = pd.concat([x_train, y_train], axis=1)
        claim = join_train[join_train['FraudFound_P']==1]
        no_claim = join_train[join_train['FraudFound_P']==0]

        undersample_noclaim = no_claim.sample(len(claim)*3)
        join_train2 = pd.concat([claim,undersample_noclaim]).sample(frac=1)

        X = join_train2.drop(columns='FraudFound_P')
        y = join_train2['FraudFound_P']
    return X, y

def metric_result(y_test, y_pred):
    print("F1 Score : ",f1_score(y_test, y_pred, average='binary'))
    print("Recall Score : ",recall_score(y_test, y_pred))
    print("Precision Score : ",precision_score(y_test, y_pred))

    ig, ax = plt.subplots(1,2, figsize=(10,5))
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax[0])
    ax[0].set_title('Confusion Matrix')

    # ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    ax[1].plot(fpr,tpr, label="AUC="+str(auc))
    ax[1].set_title('ROC AUC')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].legend(loc=4)
    plt.show()

def glm_result(res, X_test, y_test):
    X_test_sm = sm.add_constant(X_test)
    y_pred = res.predict(X_test_sm)

    df_res = pd.DataFrame({'is_claim_real':y_test, 'is_claim_prob':y_pred})
    numbers = [float(x)/10 for x in range(10)]
    for i in numbers:
        df_res[i]= df_res.is_claim_prob.map(lambda x: 1 if x > i else 0)
    cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

    cut_off = [0.0,0.05, 0.1, 0.15 , 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    f1_result = []
    for i in cut_off:
        df_res['final_predicted'] = df_res['is_claim_prob'].map( lambda x: 1 if x > i else 0)
        f1_result.append(roc_auc_score(df_res['is_claim_real'], df_res['final_predicted']))
    df_res2 = pd.DataFrame({'cut_off':cut_off, 'f1_score':f1_result})
    best_tresh = df_res2.sort_values('f1_score', ascending=False).head(1)['cut_off'].values[0]
    y_pred_thresh = (y_pred >= best_tresh).astype('float')
    print('Best Threshold :', best_tresh)
    metric_result(y_test, y_pred_thresh)

def find_best_tresh(pred_proba, y_test):
    df_res = pd.DataFrame({'prob':pred_proba})
    cut_off = [0.0,0.05, 0.1, 0.15 , 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    f1_result = []
    for i in cut_off:
        df_res['final_predicted'] = df_res['prob'].map( lambda x: 1 if x > i else 0)
        f1_result.append(roc_auc_score(y_test, df_res['final_predicted']))
    df_res2 = pd.DataFrame({'cut_off':cut_off, 'f1_score':f1_result})
    best_tresh = df_res2.sort_values('f1_score', ascending=False).head(1)['cut_off'].values[0]
    y_pred_thresh = (y_pred >= best_tresh).astype('float')
    print('Best Threshold :', best_tresh)
    metric_result(y_test, y_pred_thresh)


# # Oversampling

# In[65]:


oversample = SMOTE(sampling_strategy=0.5)
X_over, y_over = oversample.fit_resample(X_train, y_train)
print('After Oversampling:\n',y_over.value_counts())


# In[66]:


fit_model(X_over, X_test, y_over, y_test, 'over')


# # Undersampling

# In[67]:


join_train = pd.concat([X_train, y_train], axis=1)
claim = join_train[join_train['FraudFound_P']==1]
no_claim = join_train[join_train['FraudFound_P']==0]

undersample_noclaim = no_claim.sample(len(claim)*2)
join_train2 = pd.concat([claim,undersample_noclaim]).sample(frac=1)

print("Before Undersample:\n", join_train['FraudFound_P'].value_counts())
print("After Undersample:\n", join_train2['FraudFound_P'].value_counts())

X_under = join_train2.drop(columns='FraudFound_P')
y_under = join_train2['FraudFound_P']


# In[68]:


fit_model(X_under, X_test, y_under, y_test, 'Under')


# # Without Over/Under Sampling

# In[69]:


fit_model(X_train, X_test, y_train, y_test, 'Normal')


# In[70]:


df_model = pd.DataFrame({'model':modelname, 'accuracy':acc, 'recall':recall, 'precision':precision, 'f1':f1, 'roc_auc':roc_auc})


# In[71]:


df_model


# In[72]:


max_accuracy = df_model['accuracy'].max()
best_model = df_model.loc[df_model['accuracy'] == max_accuracy, 'model'].values[0]
best_model


# In[73]:


max_recall = df_model['recall'].max()
best_model = df_model.loc[df_model['recall'] == max_recall, 'model'].values[0]
best_model


# In[74]:


max_precision = df_model['precision'].max()
best_model = df_model.loc[df_model['precision'] == max_precision, 'model'].values[0]
best_model


# In[75]:


max_f1 = df_model['f1'].max()
best_model = df_model.loc[df_model['f1'] == max_f1, 'model'].values[0]
best_model


# In[76]:


max_roc_auc = df_model['roc_auc'].max()
best_model = df_model.loc[df_model['roc_auc'] == max_roc_auc, 'model'].values[0]
best_model


#  # HyperParameter Tuning Model 1 

# In[77]:


X_over, y_over = sampling(X_train, y_train, 'over', 0.5)
X_under, y_under = sampling(X_train, y_train, 'under', 0.5)


# #

# #  1 XGBoost

# In[78]:


params = {
            'n_estimators':[300], #300
          'max_depth':[7], #7
          'learning_rate':[0.1] #0.1
         }
skf = RepeatedStratifiedKFold(n_splits=3)
xgb = XGBClassifier(use_label_encoder=False, objective= 'binary:logistic',eval_metric='auc')

grid_search = GridSearchCV(estimator=xgb, param_grid=params, scoring='f1', cv=skf)
grid_search.fit(X_under, y_under)


# In[79]:


grid_search.best_score_


# In[80]:


feat_importances = pd.Series(grid_search.best_estimator_.feature_importances_, index=X_train.columns)
feat_importances.sort_values(ascending=False).head(10).plot(kind='barh')


# In[81]:


y_pred = grid_search.predict(X_test)
metric_result(y_test, y_pred)


# # 2 LightGBM

# In[82]:


params = {
          'boosting_type':['gbdt'],#300
          'n_estimators':[150],
          'num_leaves':[32], #7
          'learning_rate':[0.02] #0.1
         }
skf = RepeatedStratifiedKFold(n_splits=3)
lgbm = LGBMClassifier(objective= 'binary', class_weight='balanced')

grid_search_lgbm = GridSearchCV(estimator=lgbm, param_grid=params, scoring='f1', cv=skf)
grid_search_lgbm.fit(X_under, y_under)


# In[83]:


grid_search_lgbm.best_score_


# In[84]:


grid_search_lgbm.best_params_


# In[85]:


feat_importances = pd.Series(grid_search_lgbm.best_estimator_.feature_importances_, index=X_train.columns)
feat_importances.sort_values(ascending=False).head(10).plot(kind='barh')


# In[86]:


y_pred = grid_search_lgbm.predict(X_test)
metric_result(y_test, y_pred)


# #  3 Decision Tree Classifier

# In[87]:


params = {
          'criterion':['gini', 'entropy', 'log_loss'],#300
          'splitter':['best','random'],
          'min_samples_split':[2,3,4,5],
          'max_features':['','auto', 'sqrt', 'log2']
         }
skf = RepeatedStratifiedKFold(n_splits=3)
dt = DecisionTreeClassifier(class_weight='balanced')

grid_search_dt = GridSearchCV(estimator=dt, param_grid=params, scoring='f1', cv=skf)
grid_search_dt.fit(X_under, y_under)


# In[88]:


grid_search_dt.best_params_


# In[89]:


grid_search_dt.best_score_


# In[90]:


feat_importances = pd.Series(grid_search_dt.best_estimator_.feature_importances_, index=X_train.columns)
feat_importances.sort_values(ascending=False).head(10).plot(kind='barh')


# In[91]:


y_pred = grid_search_dt.predict(X_test)
metric_result(y_test, y_pred)


# # 4. Random Forest

# In[92]:


params = {
          'n_estimators':[200],
          'criterion':['gini'],
          'min_samples_split':[3],
          'min_samples_leaf':[3],
         }
skf = RepeatedStratifiedKFold(n_splits=3)
rf = RandomForestClassifier(class_weight='balanced')

grid_search_rf = GridSearchCV(estimator=rf, param_grid=params, scoring='f1', cv=skf)
grid_search_rf.fit(X_under, y_under)


# In[93]:


grid_search_rf.best_params_


# In[94]:


grid_search_rf.best_score_


# In[95]:


feat_importances = pd.Series(grid_search_rf.best_estimator_.feature_importances_, index=X_train.columns)
feat_importances.sort_values(ascending=False).head(10).plot(kind='barh')


# In[96]:


y_pred = grid_search_rf.predict(X_test)
metric_result(y_test, y_pred)


# # Hypertuning Model With Feature Selection From Chi2 test.

# In[97]:


#X_train2 = X_train.drop(columns=['PoliceReportFiled', 'Days_Policy_Claim', 'DayOfWeek', 'WitnessPresent', 'WitnessPresent', 'WeekOfMonthClaimed', 'DayOfWeekClaimed', 'DriverRating', 'WeekOfMonth', 'NumberOfCars', 'RepNumber'])
#X_test2 = X_test.drop(columns=['PoliceReportFiled', 'Days_Policy_Claim', 'DayOfWeek', 'WitnessPresent', 'WitnessPresent', 'WeekOfMonthClaimed', 'DayOfWeekClaimed', 'DriverRating', 'WeekOfMonth', 'NumberOfCars', 'RepNumber'])


# In[98]:


X_train2 = X_train.drop(columns=['Days_Policy_Claim','DayOfWeek','WitnessPresent','WeekOfMonthClaimed','DayOfWeekClaimed','DriverRating','WeekOfMonth','NumberOfCars','RepNumber','Make_Accura','Make_Ferrari','Make_Lexus','Make_Porche','Make_Jaguar','Make_BMW','Make_Toyota','Make_Nisson','Make_Mercury','Make_Mecedes','Make_Chevrolet','Make_Honda','Make_Ford','Make_Saturn','Make_Pontiac','Make_Dodge','Make_Saab','Make_Mazda','Make_VW'])
X_test2 = X_test.drop(columns=['Days_Policy_Claim','DayOfWeek','WitnessPresent','WeekOfMonthClaimed','DayOfWeekClaimed','DriverRating','WeekOfMonth','NumberOfCars','RepNumber','Make_Accura','Make_Ferrari','Make_Lexus','Make_Porche','Make_Jaguar','Make_BMW','Make_Toyota','Make_Nisson','Make_Mercury','Make_Mecedes','Make_Chevrolet','Make_Honda','Make_Ford','Make_Saturn','Make_Pontiac','Make_Dodge','Make_Saab','Make_Mazda','Make_VW'])


# In[99]:


# X_over, y_over = sampling(X_train, y_train, 'over', 0.5)
X_under2, y_under2 = sampling(X_train2, y_train, 'under')


# # 1. XGBoost

# In[100]:


params = {
            'n_estimators':[300], #300
          'max_depth':[8], #8
          'learning_rate':[0.1] #0.1
         }
skf = RepeatedStratifiedKFold(n_splits=3)
xgb = XGBClassifier(use_label_encoder=False, objective= 'binary:logistic',eval_metric='auc')

grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=params, scoring='f1', cv=skf)
grid_search_xgb.fit(X_under2, y_under2)


# In[101]:


grid_search_xgb.best_params_


# In[102]:


grid_search_xgb.best_score_


# In[103]:


feat_importances = pd.Series(grid_search_xgb.best_estimator_.feature_importances_, index=X_train2.columns)
feat_importances.sort_values(ascending=False).head(10).plot(kind='barh')


# In[104]:


y_pred = grid_search_xgb.predict(X_test2)
metric_result(y_test, y_pred)


# # 2. LightGBM

# In[105]:


params = {
          'boosting_type':['gbdt'],
          'n_estimators':[300], #300
          'num_leaves':[32], #32
          'learning_rate':[0.01] #0.01
         }
skf = RepeatedStratifiedKFold(n_splits=3)
lgbm = LGBMClassifier(objective= 'binary', class_weight='balanced')

grid_search_lgbm = GridSearchCV(estimator=lgbm, param_grid=params, scoring='f1', cv=skf)
grid_search_lgbm.fit(X_under2, y_under2)


# In[106]:


grid_search_lgbm.best_params_


# In[107]:


grid_search_lgbm.best_score_


# In[108]:


feat_importances = pd.Series(grid_search_lgbm.best_estimator_.feature_importances_, index=X_train2.columns)
feat_importances.sort_values(ascending=False).head(10).plot(kind='barh')


# In[109]:


y_pred = grid_search_lgbm.predict(X_test2)
metric_result(y_test, y_pred)


# # 3. Decision Tree

# In[110]:


params = {
          'criterion':['entropy'],#entropy
          'splitter':['best'],
          'min_samples_split':[4], #4
          'max_features':['auto'] #auto
         }
skf = RepeatedStratifiedKFold(n_splits=3)
dt = DecisionTreeClassifier(class_weight='balanced')

grid_search_dt = GridSearchCV(estimator=dt, param_grid=params, scoring='f1', cv=skf)
grid_search_dt.fit(X_under2, y_under2)


# In[111]:


grid_search_dt.best_params_


# In[112]:


grid_search_dt.best_score_


# In[113]:


feat_importances = pd.Series(grid_search_dt.best_estimator_.feature_importances_, index=X_train2.columns)
feat_importances.sort_values(ascending=False).head(10).plot(kind='barh')


# In[114]:


y_pred = grid_search_dt.predict(X_test2)
metric_result(y_test, y_pred)


# # 4. Random Forest

# In[115]:


params = {
          'n_estimators':[300], #300
          'criterion':['entropy'], #entropy
          'min_samples_split':[2], # 2
          'min_samples_leaf':[4], #4
         }
skf = RepeatedStratifiedKFold(n_splits=3)
rf = RandomForestClassifier(class_weight='balanced')

grid_search_rf = GridSearchCV(estimator=rf, param_grid=params, scoring='f1', cv=skf)
grid_search_rf.fit(X_under2, y_under2)


# In[116]:


grid_search_rf.best_score_


# In[117]:


grid_search_rf.best_params_


# In[118]:


feat_importances = pd.Series(grid_search_rf.best_estimator_.feature_importances_, index=X_train2.columns)
feat_importances.sort_values(ascending=False).head(10).plot(kind='barh')


# In[119]:


y_pred = grid_search_rf.predict(X_test2)
metric_result(y_test, y_pred)


# # SHAP XGBoost

# In[120]:


import shap


# In[121]:


explainer = shap.TreeExplainer(grid_search_xgb.best_estimator_, X_under2)
shap_values = explainer.shap_values(X_under2)
shap.initjs()
shap.summary_plot(shap_values, X_under2, plot_type = "bar")


# In[147]:


fig2=shap.summary_plot(shap_values, X_under2, plot_type="layered_violin", show = False)
plt.savefig('shap_standard_plot.png', bbox_inches='tight', dpi=300)


# In[124]:


shap.force_plot(explainer.expected_value, shap_values[2, :], X_under2.iloc[2, :])


# In[125]:


shap.force_plot(explainer.expected_value, shap_values[8, :], X_under2.iloc[8, :])


# In[126]:


shap.force_plot(explainer.expected_value, shap_values[15, :], X_under2.iloc[15, :])


# In[127]:


shap.force_plot(explainer.expected_value, shap_values[0, :], X_under2.iloc[0, :])


# In[128]:


shap.force_plot(explainer.expected_value, shap_values[1, :], X_under2.iloc[1, :])


# In[129]:


shap.force_plot(explainer.expected_value, shap_values[6, :], X_under2.iloc[6, :])


# # Saving Plots.

# In[130]:


# Using 'show = False' and 'matplotlib = True' overwrites java and uses matplotlib to save the figures as png
#shap.force_plot(explainer.expected_value, shap_values[15, :], X_under2.iloc[15, :] , show = False, matplotlib = True)
#plt.savefig('Fraud3.png',format = "png",dpi = 150,bbox_inches = 'tight')


# In[131]:


y_under2.head(20)


# In[132]:


X_under2.head(11)


# # LIME XGBoost

# In[133]:


import lime
from lime import lime_tabular


# In[134]:


np.random.seed(123)
predict_fn = lambda x: grid_search_xgb.predict_proba(x)


# In[135]:


explainer2 = lime_tabular.LimeTabularExplainer(
    training_data = np.array(X_under2),
    feature_names = X_under2.columns,
    class_names=['Not Fraud', 'Fraud'],
    mode = 'classification'
)


# In[136]:


data_row = X_under2.iloc[2] # iloc[0] is the first fraud detected by the model.


# In[145]:


y_under2.head(30)


# In[144]:


exp = explainer2.explain_instance(
    data_row=data_row.values, 
    predict_fn=predict_fn,
    num_features=20
)
# Generate the explanation figure
exp.show_in_notebook(show_table=True)



# In[146]:


data_row = X_under2.iloc[28] # iloc[0] is the first fraud detected by the model.
exp = explainer2.explain_instance(
    data_row=data_row.values, 
    predict_fn=predict_fn,
    num_features=10
)
exp.show_in_notebook(show_table=True)
exp.save_to_file('LIMEfraud2.html')


# In[140]:


data_row = X_under2.iloc[15] # iloc[0] is the first fraud detected by the model.
exp = explainer2.explain_instance(
    data_row=data_row.values, 
    predict_fn=predict_fn,
    num_features=10
)
exp.show_in_notebook(show_table=True)
exp.save_to_file('LIMEfraud3.html')


# In[141]:


data_row = X_under2.iloc[0] # iloc[0] is the first fraud detected by the model.
exp = explainer2.explain_instance(
    data_row=data_row.values, 
    predict_fn=predict_fn,
    num_features=10
)
exp.show_in_notebook(show_table=True)
exp.save_to_file('LIME_Nfraud1.html')


# In[142]:


data_row = X_under2.iloc[1] # iloc[0] is the first fraud detected by the model.
exp = explainer2.explain_instance(
    data_row=data_row.values, 
    predict_fn=predict_fn,
    num_features=10
)
exp.show_in_notebook(show_table=True)
exp.save_to_file('LIME_Nfraud2.html')


# In[143]:


data_row = X_under2.iloc[6] # iloc[0] is the first fraud detected by the model.
exp = explainer2.explain_instance(
    data_row=data_row.values, 
    predict_fn=predict_fn,
    num_features=10
)
exp.show_in_notebook(show_table=True)
exp.save_to_file('LIME_Nfraud3.html')

