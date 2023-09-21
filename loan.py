#!/usr/bin/env python
# coding: utf-8

# In[23]:


# Data analysis and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns

# classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[24]:


import pandas as pd     
df=pd.read_csv(r"D:\New folder\train.csv")
print(df)


# In[25]:


df.info()


# In[26]:


print(df.size)
print(df.columns)
print(df.head())


# In[27]:


df.dtypes


# In[28]:


df.describe()


# In[29]:


# Visualize the missing data
import seaborn as sns
plt.figure(figsize=(10,6))
sns.displot(
    data=df.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.25)
plt.show()


# In[30]:


# count of missing values
df.isna().sum()


# In[9]:


#  % of missing values
df.isna().sum()*100 /len(df)


# In[10]:


#handle numerical missing data
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())


# In[11]:


#handle categorial missing data
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])


# In[12]:


# again check null values
df.isna().sum()


# In[13]:


# Preview the data again
df.head()


# In[14]:


import seaborn as sns
sns.countplot(df['Gender'])


# In[15]:


sns.countplot(df['Dependents'])


# In[16]:


#Plot4- Scatterplot
fig, ax = plt.subplots(2,2, figsize=(14,12))

sns.scatterplot(data=df,x="ApplicantIncome", y="LoanAmount",s=70, hue="Loan_Status", palette='ocean',ax=ax[0,0])
sns.histplot(df, x=df['LoanAmount'], bins=10, ax=ax[0,1])
sns.scatterplot(data=df,x='CoapplicantIncome', y='LoanAmount',s=70, hue='Loan_Status',palette='ocean', ax=ax[1,0])
sns.scatterplot(data=df,x='Loan_Amount_Term', y='LoanAmount', s=70, hue='Loan_Status',palette='ocean', ax=ax[1,1])

plt.show()


# In[17]:


# another preview of the data
df.head()


# In[18]:


#identify all categorical columns & pass into a variable
objectlist_train = df.select_dtypes(include = "object").columns

#Label Encoding for object to numeric conversion

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feature in objectlist_train:
    df[feature] = le.fit_transform(df[feature].astype(str))

print (df.info())


# In[19]:


# repeat the same process to encode the test data
objectlist_test = df.select_dtypes(include='object').columns

for feature in objectlist_test:
    df[feature] = le.fit_transform(df[feature].astype(str))

print (df.info())


# In[20]:


x = df.iloc[:,1:].drop('Loan_Status', axis=1) 
# drop loan_status column because that is what we are predicting
y = df['Loan_Status']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.30, random_state=0)


# In[21]:


df_model = DecisionTreeClassifier()
df_model.fit(train_x, train_y)
predict_y = df_model.predict(test_x)
print(classification_report(test_y, predict_y))
print("Accuracy:", accuracy_score(predict_y, test_y))


# In[22]:


# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(train_x, train_y)
predict_y_2 = rf_model.predict(test_x)
print(classification_report(test_y, predict_y_2))
print("Accuracy:", accuracy_score(predict_y_2, test_y))

