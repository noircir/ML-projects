#!/usr/bin/env python
# coding: utf-8

# ## Blind dataset

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df = pd.read_csv('Classified Data', index_col=0)


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df.describe()


# ### Standardize data for KNN

# In[10]:


# For KNN it is better to standardize data. Because the scale matters (neighbours).

from sklearn.preprocessing import StandardScaler


# In[11]:


scaler = StandardScaler()


# In[12]:


scaler.fit(df.drop('TARGET CLASS', axis=1))


# In[13]:


scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))


# In[21]:


scaled_features[:5]


# In[28]:


# The last element [-1] is going to be the target
df_features = pd.DataFrame(scaled_features,columns=df.columns[:-1])


# In[29]:


df_features.head()


# In[31]:


from sklearn.model_selection import train_test_split


# In[33]:


X = df_features
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.3, random_state=101,)

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[34]:


from sklearn.neighbors import KNeighborsClassifier


# In[35]:


# Only 1 neighbour !!! at first

knn = KNeighborsClassifier(n_neighbors=1)


# In[36]:


knn.fit(X_train,y_train)


# In[37]:


prediction = knn.predict(X_test)


# In[38]:


from sklearn.metrics import classification_report, confusion_matrix


# In[40]:


print(classification_report(y_test,prediction))


# In[41]:


print(confusion_matrix(y_test, prediction))


# ### What else can we squeeze for evaluation?

# In[42]:


# Let's explore different k-values. Though the metrics are already good, 
# maybe some other would be better?

# Mean 
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    # The elbow method: the sum of the total error divided by the number of records
    # ((pred_i[0] - y_test[0]) + (pred_i[1] - y_test[1]) + ... + (pred_i[n] - y_test[n])) / n
    error_rate.append(np.mean( pred_i != y_test ))


# In[45]:


error_rate


# In[50]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',
        markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[52]:


# let's pick k=17:
knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print(classification_report(y_test,pred))
print(confusion_matrix(y_test, pred))


# In[53]:


# let's pick k=12:
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print(classification_report(y_test,pred))
print(confusion_matrix(y_test, pred))


# In[ ]:




