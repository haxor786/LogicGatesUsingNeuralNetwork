
# coding: utf-8

# In[65]:


import keras
import numpy as np
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Activation, Dense


# In[3]:


data_in = np.array([[0,0],
                     [0,1],
                    [1,0],
                   [1,1]])
print (data_in)


# In[4]:


logic_and = np.array([ [0],[0] ,[0],[1]
    
])
print (logic_and)


# In[5]:


logic_or = np.array([[0],[1],[1],[1]])
print (logic_or)


# In[6]:


logic_xor = np.array([[0],[1],[1],[0]])
print (logic_xor)


# In[7]:


logic_nand = np.array([[1],[1],[1],[0]])
print (logic_nand)


# In[163]:


logic_nor = np.array([[1],[0],[0],[0]])
print (logic_nor)


# In[ ]:


##AND && OR gate


# In[188]:


model = Sequential()
model.add(Dense(1 ,input_dim=2,activation='sigmoid'))
#model.add(Dense(1,activation='sigmoid))
#model.add(Dense(1,activation='sigmoid'))
sgd = SGD(lr = 0.5)
model.compile(optimizer='sgd',loss='mse')


# In[189]:


model.fit(data_in,logic_and,epochs=1000,verbose=False)


# In[190]:


model.predict(np.array([[1,0]]))


# In[191]:


model.fit(data_in,logic_or,epochs = 1000,verbose=False)


# In[192]:


model.predict(np.array([[0,0]]))


# In[179]:



model1 = Sequential()
model1.add(Dense(2 ,input_dim=2,activation='sigmoid'))
model1.add(Dense(1,activation='sigmoid'))
#model.add(Dense(1,activation='sigmoid'))
sgd = SGD(lr = 0.5)
model1.compile(optimizer='sgd',loss='mse')


# In[184]:


model1.fit(data_in,logic_xor,epochs=1000,verbose=False)


# In[186]:


model1.predict(np.array([[1,1]]))

