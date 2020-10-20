#!/usr/bin/env python
# coding: utf-8

# <img src="./images/DLI_Header.png" style="width: 400px;">

# # Convolutional Neural Networks

# In the previous section, you built and trained a simple model to classify ASL images. If you recall, the model was able to learn how to correctly classify the training dataset with very high accuracy, but, it did not perform nearly as well on validation dataset. This behavior of not generalizing well to non-training data is called *overfitting*, and in this section, we will introduce a popular kind of model called a [convolutional neural network](https://www.youtube.com/watch?v=x_VrgWTKkiM&vl=en) that is especially good for reading images and classifying them.

# ## Objectives

# By the time you complete this section you will be able to:
# * Prep data specifically for a CNN
# * Create a more sophisticated CNN model, understanding a greater variety of model layers
# * Train a CNN model and observe its performance

# ## Loading and Preparing the Data

# So we can move more quickly on to new topics, you can execute the following cell to load and prepare the ASL dataset for training:

# In[1]:


import tensorflow.keras as keras
import pandas as pd

# Load in our data from CSV files
train_df = pd.read_csv("asl_data/sign_mnist_train.csv")
test_df = pd.read_csv("asl_data/sign_mnist_test.csv")

# Separate out our target values
y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']

# Separate out our image vectors
x_train = train_df.values
x_test = test_df.values

# Turn our scalar targets into binary categories
y_train = keras.utils.to_categorical(y_train, 25)
y_test = keras.utils.to_categorical(y_test, 25)

# Normalize our image data
x_train = x_train / 255
x_test = x_test / 255


# ## Reshaping Images for a CNN

# You may remember from the last exercise that the individual pictures in our dataset are in the format of long lists of 784 pixels:

# In[2]:


x_train.shape, x_test.shape


# In this format, we don't have all the information about which pixels are near each other. Because of this, we can't apply convolutions that will detect features. Let's reshape our dataset so that they are in a 28x28 pixel format. This will allow our convolutions to read over each image and detect important features.
# 
# Note that for the first convolutional layer of our model, we need to have not only the height and width of the image, but also the number of color channels. Our images are grayscale, so we'll just have 1 channel.
# 
# That means that we need to convert the current shape `(27455, 784)` to `(27455, 28, 28, 1)`. As a convenience, we can pass the `reshape` method a `-1` for any dimension we wish to remain the same, therefore:

# In[3]:


x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)


# In[4]:


x_train.shape


# In[5]:


x_test.shape


# In[6]:


x_train.shape, x_test.shape


# ## Creating a Convolutional Model

# We want to assure you that through the beginning of your deep learning journey, you will not have to create your own model without guidance. Assuming your problem is not totally unique, there's a great chance that people have created models that will perform well for you. For instance, it just takes a bit of googling to find a good set of layers to construct a convolutional model. Today, we'll provide you with a model that will work well for this problem.
# 
# We covered many of the different kinds of layers in the lecture, and we will go over them all here. Don't worry about memorizing them.

# In[7]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization

num_classes = 25

model = Sequential()
model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 512 , activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = num_classes , activation = 'softmax'))


# ### Conv2D

# These are our 2D convolutional layers. Small kernels will go over the input image and detect features that are important for classification. Earlier convolutions in the model will detect simple features such as lines. Later convolutions will detect more complex features. Let's look at our first Conv2D layer:
# ```Python
# model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same'...)
# ```
# 75 refers to the number of filters that will be learned. (3,3) refers to the size of those filters. Strides refer to the step size that the filter will take as it passes over the image. Padding refers to whether the output image that's created from the filter will match the size of the input image. 

# ### BatchNormalization

# Like normalizing our inputs, batch normalization scales the values in the hidden layers to improve training. If you'd like, [read more about it in detail here](https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c). 

# ### MaxPool2D

# Max pooling takes an image and essentially shrinks it to a lower resolution. It does this to help the model be robust to translation (objects moving side to side), and also makes our model faster.

# ### Dropout

# Dropout is a technique for preventing overfitting. Dropout randomly selects a subset of neurons and turns them off, so that they do not participate in forward or backward propogation in that particular pass. This helps to make sure that the network is robust and redundant, and does not rely on any one area to come up with answers.    

# ### Flatten

# Flatten takes the output of one layer which is multidimensional, and flattens it into a one-dimensional array. The output is called a feature vector and will be connected to the final classification layer.

# ### Dense

# We have seen dense layers before in our earlier models. Our first dense layer (512 units) takes the feature vector as input and learns which features will contribute to a particular classification. The second dense layer (24 units) is the final classification layer that outputs our prediction.

# ## Summarizing the Model

# This may feel like a lot of information, but don't worry. It's not critical that you understand everything right now in order to effectively train convolutional models. Most importantly you know that they can help with extracting useful information from images, and can be used in classification tasks.

# Here we summarize the model we just created:

# In[8]:


model.summary()


# ## Compiling the Model

# We'll compile the model just like before:

# In[9]:


model.compile(loss = 'categorical_crossentropy' , metrics = ['accuracy'])


# ## Training the Model

# Despite the very different model architecture, the training looks exactly the same. Run the cell below to train for 20 epochs and let's see if the accuracy improves:

# In[10]:


model.fit(x_train, y_train,
                    epochs=20,
                    verbose=1,
                    validation_data=(x_test, y_test))


# ## Discussion of Results

# It looks like this model is significantly improved! The training accuracy is very high, and the validation accuracy has improved as well. This is a great result, as all we had to do was swap in a new model.
# 
# You may have noticed the validation accuracy jumping around. This is an indication that our model is still not generalizing perfectly. Fortunately there's more that we can do. Let's talk about it in the next lecture.

# ## Summary

# In this section you utilized several new kinds of layers to implement a CNN, which performed better than the more simple model used in the last section. Hopefully the overall process of creating and training a model with prepared data is starting to become even more familiar.

# ## Clear the Memory
# Before moving on, please execute the following cell to clear up the GPU memory. This is required to move on to the next notebook.

# In[11]:


import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)


# ## Next

# In the last several sections you have focused on the creation and training of models. In order to further improve performance, you will now turn your attention to *data augmentation*, a collection of techniques that will allow your models to train on more and better data than what you might have originally at your disposal.
# 
# Please continue to the next section: [*Data Augmentation*](./04a_asl_augmentation.ipynb).
