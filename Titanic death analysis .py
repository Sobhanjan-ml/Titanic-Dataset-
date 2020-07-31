#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dataset = pd.read_csv('titanic.csv')


# In[3]:


dataset


# In[4]:


dataset.head(10)


# In[5]:


dataset.isnull().any()


# In[6]:


dataset.isna().sum()


# In[7]:


sns.set_style('whitegrid')
sns.countplot(dataset['Survived'])


# In[8]:


dataset['Survived'].value_counts()


# In[9]:


sns.countplot(dataset['Sex'])


# In[10]:


sns.countplot(x='Survived',hue='Sex',data=dataset)


# In[11]:


dataset['Age'].plot.hist(bins=40)
plt.xlabel('Age')


# In[12]:


sns.countplot(dataset['SibSp'])


# In[13]:


dataset['Fare'].plot.hist(bins=20)


# In[14]:


dataset['Age'].fillna((dataset['Age'].mean()),inplace=True)


# In[15]:


dataset


# In[16]:


dataset['Age'].round()


# In[17]:


dataset


# In[18]:


datase = dataset['Age'].round()


# In[19]:


dataset


# In[20]:


dataset.drop('Cabin',axis=1,inplace=True)


# In[21]:


dataset.head()


# In[22]:


dataset.dropna(inplace=True)
dataset


# In[23]:


dataset.isnull().any()


# In[24]:


from sklearn.preprocessing import LabelEncoder


# In[25]:


lb = LabelEncoder()


# In[26]:


dataset['Sex'] = lb.fit_transform(dataset['Sex'])


# In[27]:


dataset


# In[28]:


dataset.drop(['Name','Ticket'],axis=1,inplace=True)


# In[29]:


dataset


# In[30]:


x = dataset.iloc[:,2:].values


# In[31]:


x


# In[32]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[33]:


ct = ColumnTransformer([('oh',OneHotEncoder(),[6])],remainder='passthrough')


# In[34]:


x = ct.fit_transform(x)


# In[35]:


x


# In[36]:


x = x[:,1:]


# In[37]:


x


# In[40]:


y = dataset.iloc[:,1].values


# In[41]:


y


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[44]:


x_train


# In[45]:


x_test


# In[46]:


from sklearn.linear_model import LogisticRegression


# In[47]:


logr = LogisticRegression()


# In[48]:


logr.fit(x_train,y_train)


# In[50]:


y_pred = logr.predict(x_test)
y_pred


# In[51]:


from sklearn.metrics import accuracy_score


# In[53]:


accuracy_score(y_test,y_pred)*100


# In[56]:


logr.predict([[1.0,0.0,3,1,22.0,1,0,7.2500]])


# In[ ]:




