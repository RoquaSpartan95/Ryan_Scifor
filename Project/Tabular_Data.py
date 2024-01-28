#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


# In[2]:


us_babies = pd.read_csv("us_baby_names.csv")


# In[3]:


us_babies


# In[6]:


sorted_us_babies = us_babies.sort_values('Count',ascending =False)


# In[7]:


sorted_us_babies


# In[10]:


ac=sorted_us_babies.head()


# In[11]:


ac.boxplot(by ='Name', column =['Count'], grid = False)


# In[ ]:


us_babies['Year']


# In[9]:


us_babies['Year']==2014


# In[10]:


us_babies_2014 =us_babies.loc[us_babies['Year']==2014,:]


# In[11]:


us_babies_2014


# In[12]:


sorted_us_2014 = us_babies_2014.sort_values('Count',ascending =False)


# In[13]:


sorted_us_2014


# In[17]:


most_popular_babies_names = sorted_us_2014.head()


# In[18]:


most_popular_babies_names


# In[21]:


sn.barplot(x='Name',y='Count', data = most_popular_babies_names,ci = False)
plt.show()


# In[15]:


us_Male=us_babies['Gender']=='M'


# In[16]:


us_Male


# In[18]:


us_babies_males =us_babies.loc[us_Male,:]


# In[19]:


us_babies_males


# In[22]:


popular_males=us_babies_males.sort_values('Count',ascending =False)


# In[23]:


popular_males


# In[32]:


ab=popular_males.head()


# In[33]:


ab


# In[34]:


x=ab["Name"]
y=ab["Count"]
plt.pie(y,labels=x,radius=1.2,shadow=True)
plt.show()


# In[20]:


us_babies_females=us_babies.loc[us_babies['Gender']=='F',:]


# In[21]:


us_babies_females


# In[25]:


popular_females=us_babies_females.sort_values('Count',ascending =False)


# In[26]:


popular_females


# In[29]:


pfh=popular_females.head()


# In[30]:


pfh


# In[31]:


x=pfh["Name"]
y=pfh["Count"]
plt.pie(y,labels=x,radius=1.2,shadow=True)
plt.show()


# In[3]:


us_babies_2008 =us_babies.loc[us_babies['Year']==2008,:]


# In[4]:


us_babies_2008


# In[5]:


popular_usf_2008=us_babies_2008.loc[us_babies['Gender']=='F',:]


# In[6]:


popular_usf_2008


# In[7]:


Popular_female_year08=popular_usf_2008.head()


# In[8]:


Popular_female_year08


# In[9]:


x=Popular_female_year08["Name"]
y=Popular_female_year08["Count"]
plt.pie(y,labels=x,radius=1.2,shadow=True)
plt.show()


# In[10]:


popular_usM_08=us_babies_2008.loc[us_babies['Gender']=='M',:]


# In[11]:


Most_PopularM_yr08=popular_usM_08.head()


# In[12]:


Most_PopularM_yr08


# In[14]:


x=Most_PopularM_yr08["Name"]
y=Most_PopularM_yr08["Count"]
plt.pie(y,labels=x,radius=1.2,shadow=True)
plt.show()


# In[ ]:




