#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


mnist = tf.keras.datasets.mnist


# In[3]:


(training_data, training_labels), (test_data, test_labels) = mnist.load_data()


# In[4]:


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[5]:


model.fit(training_data,training_labels, epochs=5)


# In[6]:


model.evaluate(test_data,test_labels)


# In[7]:


model.save("num_reader.model")


# In[8]:


new_model = tf.keras.models.load_model("num_reader.model")


# In[9]:


predictions = new_model.predict(test_data)


# In[10]:


print(predictions)


# In[11]:


print(np.argmax(predictions[0]))


# In[19]:


plt.imshow(test_data[0],cmap=plt.cm.binary)
plt.show()


# In[ ]:




