#!/usr/bin/env python
# coding: utf-8

# ## Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Importing the dataset

# In[2]:


df = pd.read_csv('/Users/ashleshad/Downloads/breast_cancer.csv')


# ## EDA

# In[3]:


df.head()


# In[4]:


df.describe()


# In[33]:


df['Class'].value_counts()


# #### Values :
#     4 - refer to ‘malignant’ (likely to have breast cancer) or 
#     2 - refer to ‘benign’ (likely doesn't breast cancer)
#     Let's replace 2 with 0 and 4 with 1.

# In[34]:


df['Class'].replace(2,0,inplace=True)
df['Class'].replace(4,1,inplace=True)


# In[35]:


df.head()


# In[36]:


correlation_matrix = df.corr()

plt.figure(figsize=(20, 10))

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)

# Adding title
plt.title('Correlation Matrix Heatmap')

# Show the plot
plt.show()


# In[ ]:





# In[37]:


df.isnull().sum()


# ## Splitting into training and test set

# In[38]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# In[39]:


X = df.drop('Class',1)


# In[40]:


X


# In[41]:


Y = df['Class']


# In[42]:


Y


# ## Performing by Logistic Regression

# In[43]:


LR = LogisticRegression()


# In[44]:


from sklearn.model_selection import train_test_split


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state=4)


# In[46]:


LR.fit(X_train, y_train)


# In[47]:


model_predict=LR.predict(X_test)


# In[48]:


LR.score(X_train,y_train)


# In[49]:


LR.score(X_test,y_test)


# In[50]:


print(classification_report(y_test, model_predict))


# ## Performing by Decision Tree Classifier

# In[51]:


from sklearn.tree import DecisionTreeClassifier


# In[52]:


DTC = DecisionTreeClassifier(criterion='entropy', max_depth=2)


# In[53]:


DTC.fit(X_train,y_train)


# In[54]:


model_pred=DTC.predict(X_test)


# In[55]:


model_pred


# In[56]:


print(classification_report(y_test,model_pred))


# ## Performing by Random Forest Classifier 

# In[57]:


from sklearn.ensemble import RandomForestClassifier


# In[58]:


RFC=RandomForestClassifier(n_estimators=100, random_state=4)


# In[59]:


RFC.fit(X_train,y_train)


# In[60]:


y_pred = RFC.predict(X_test)


# In[61]:


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report


# In[62]:


confusion_matrix(y_test,y_pred)


# In[63]:


print(classification_report(y_test,y_pred))


# ### Logistic Regression and Random Forest both have performed better than the Decision tree with accuracy of 98% based on the provided evaluation metrics. However, if interpretability is a priority, then preferring logistic regression over random forest would make sense. Logistic regression provides interpretable coefficients for each feature, making it easier to understand the impact of each variable on the prediction. 

# In[ ]:




