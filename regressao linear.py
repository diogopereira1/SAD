
# coding: utf-8

# In[1]:

import pandas as pd


# In[3]:

# instacia uma variavel com o banco de dados
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

# mostra a 5 primeiras linhas
data.head()


# In[4]:

# mostra as 5 ultimas linhas

data.tail()


# In[6]:

# numero total de linhas e colunas na tabela
data.shape


# In[7]:

import seaborn as sns

get_ipython().magic('matplotlib inline')


# In[8]:

sns.pairplot(data, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', size=7, aspect=0.7, kind='reg')


# In[10]:

x=data[['TV', 'radio', 'newspaper']]

x.head()


# In[19]:

y=data['sales']

y.head()


# In[23]:

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)


# In[25]:

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[26]:

from sklearn.linear_model import LinearRegression

linreg=LinearRegression()

linreg.fit(x_train, y_train)


# In[27]:

print(linreg.intercept_)
print(linreg.coef_)


# In[28]:

list(zip(features_cols, linreg.coef_))


# In[29]:

y_pred=linreg.predict(x_test)


# In[30]:

true = [100,50, 30, 20]
pred = [90, 50, 50, 30]


# In[40]:

from sklearn import metrics as mt
import numpy as np

print(mt.mean_squared_error(y_test, y_pred))

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[41]:

x=data[['TV', 'radio']]
y=data.sales

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

linreg.fit(x_train, y_train)

y_pred=linreg.predict(x_test)

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:



