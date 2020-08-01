#!/usr/bin/env python
# coding: utf-8

# In[75]:


import numpy as np


# In[5]:


import pandas as pd
import sklearn
import matplotlib.pyplot as plt


# In[6]:


le_details = pd.read_csv("IceCreamData.csv")


# In[7]:


le_details.shape


# In[8]:


le_details


# In[38]:


x = le_details.drop('Revenue',axis=1)
x


# In[39]:


y = le_details.drop('Temperature',axis =1)
y


# In[40]:


plt.scatter(x,y)
plt.show()


# In[41]:


from sklearn.model_selection import train_test_split


# In[43]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train.shape


# In[44]:


from sklearn.linear_model import LinearRegression as lm


# In[57]:


model_trainer = lm()
model =model_trainer.fit(x_train,y_train)


# In[58]:


model.coef_ #gives value of slope


# In[63]:


y_predict = model.predict(x_test)


# In[81]:


y_new_test = np.array((y_test))
y_new_predict = np.array((y_predict))
#y_new_test.content
y_new_predict.shape
y_new_test.shape


# In[82]:


df = pd.DataFrame({'Actual': y_new_test.flatten(), 'Predicted': y_new_predict.flatten()})


# In[83]:


df


# In[88]:


plt.scatter(x_test,y_test,color = "blue")
plt.plot(x_test,y_predict, color = "red",linewidth = "3")
plt.show()


# In[89]:


df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:




