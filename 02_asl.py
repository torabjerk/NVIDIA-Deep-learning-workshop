#!/usr/bin/env python
# coding: utf-8

# <img src="./images/DLI_Header.png" style="width: 400px;">

# # Image Classification of an American Sign Language Dataset

# In this section, you will perform the data preparation, model creation, and model training steps you observed in the last section using a different dataset: images of hands making letters in [American Sign Language](http://www.asl.gs/).

# ## Objectives

# By the time you complete this section you will be able to:
# * Prepare image data for training
# * Create and compile a simple model for image classification
# * Train an image classification model and observe the results

# ## American Sign Language Dataset

# The [American Sign Language alphabet](http://www.asl.gs/) contains 26 letters. Two of those letters (j and z) require movement, so they are not included in the training dataset.  

# <img src="./images/asl.png" style="width: 600px;">

# ### Kaggle

# This dataset is available from the website [Kaggle](http://www.kaggle.com), which is a fantastic place to find datasets and other deep learning resources. In addition to providing many fantastic resources like datasets and "kernels" that are like these notebooks, Kaggle hosts competitions that you can take part in, competing with others in training highly accurate models.
# 
# If you're looking to practice or see examples of many deep learning projects, Kaggle is a great site to visit.

# ## Loading the Data

# We are going to walk you through loading the ASL dataset since it is not available via Keras in the same way that MNIST was. By the end of this section we will have `x_train`, `y_train`, `x_test`, and `y_test` variables familiar to you from the last section.

# ### Reading in the Data

# The sign language dataset is in [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) (Comma Separated Values) format. You may be familiar with CSV if you've worked with spreadsheets before. Essentially it is just a grid of rows and columns with labels at the top. 
# 
# To load and work with the data, we'll be using a library called [Pandas](https://pandas.pydata.org/), which is a highly performant tool for loading and manipulating data. We'll read the CSV files into a format called a Pandas Dataframe, which is how Pandas stores grids of data.

# In[1]:


import pandas as pd


# Pandas has a `read_csv` method that expects a file to a csv file, and returns a dataframe:

# In[2]:


train_df = pd.read_csv("asl_data/sign_mnist_train.csv")
test_df = pd.read_csv("asl_data/sign_mnist_test.csv")


# ### Exploring the Data

# Let's take a look at our data. We can use the [`head`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.head.html) method to print the first few rows of the dataframe. As you can see, each row is an image which has a `label` column, and also, 784 values representing each pixel value in the image, just like with the MNIST dataset. Note that the labels currently are numerical values, not letters of the alphabet:

# In[3]:


train_df.head()


# ### Extracting the Labels

# As with MNIST, we would like to store our training and testing labels in `y_train` and `y_test` variables. Here we create those variables and then delete the labels from our original dataframes, where they are no longer needed:

# In[4]:


y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']


# ### Extracting the Images

# As with MNIST, we would like to store our training and testing images in `x_train` and `x_test` variables. Here we create those variables:

# In[5]:


x_train = train_df.values
x_test = test_df.values


# ### Summarizing the Training and Testing Data

# As you can see we now have 27,455 images with 784 pixels each for training...

# In[6]:


x_train.shape


# ...as well as their corresponding labels:

# In[7]:


y_train.shape


# For testing, we have 7,172 images...

# In[8]:


x_test.shape


# ...and their corresponding labels:

# In[9]:


y_test.shape


# ## Visualizing the Data

# To visualize the images we will again use the matplotlib library. You don't need to worry about the details of this visualization, but if you wish can learn more about [matplotlib](https://matplotlib.org/) at a later time. Note that we'll have to reshape the data from its current 1D shape of 784 pixels, to a 2D shape of 28x28 pixels to make sense of the image:

# In[10]:


import matplotlib.pyplot as plt
plt.figure(figsize=(40,40))

num_images = 20
for i in range(num_images):
    row = x_train[i]
    label = y_train[i]
    
    image = row.reshape(28,28)
    plt.subplot(1, num_images, i+1)
    plt.title(label, fontdict={'fontsize': 30})
    plt.axis('off')
    plt.imshow(image, cmap='gray')


# ## Exercise: Normalize the Image Data

# As we did with the MNIST dataset, we are going to normalize the image data, meaning that their pixel values, instead of being between 0 and 255 as they are currently:

# In[11]:


x_train.min()


# In[12]:


x_train.max()


# ...should be floating point values between 0 and 1. Use the following cell to work. If you get stuck, look at the solution below.

# In[13]:


x_train = x_train/255
x_test = x_test/255


# ### Solution

# Click on the '...' below to show the solution.

# ```python
# x_train = x_train / 255
# x_test = x_test / 255
# ```

# ## Exercise: Categorize the Labels

# As we did with the MNIST dataset, we are going to categorically encode the labels. Recall that you can use the `keras.utils.to_categorical` method to accomplish this by passing it the values you wish to encode, and, the number of categories you would like to encode it into. Do your work in the cell below. We have imported `keras` and set the number of categories (25) for you.

# In[14]:


import tensorflow.keras as keras
num_classes = 25


# In[15]:


y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)


# ### Solution

# Click on the '...' below to show the solution.

# ```python
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
# ```

# ## Exercise: Build the Model

# The data is all prepared, you have normalized images for training and testing, as well as categorically encoded labels for training and testing.
# 
# For this exercise you are going to build a sequential model. Just like last time build a model that:
# * Has a dense input layer. This layer should contain 512 neurons, use the `relu` activation function, and expect input images with a shape of `(784,)`
# * Has a second dense layer with 512 neurons which uses the `relu` activation function
# * Has a dense output layer with neurons equal to the number of classes, using the `softmax` activation function
# 
# Do your work in the cell below, creating a `model` variable to store the model you build. We've imported the Keras `Sequental` model class and `Dense` layer class to get you started. Reveal the solution below if you get stuck:

# In[18]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[19]:


model = Sequential()

model.add(Dense(units=512, activation='relu', input_shape=(784,)))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units = 25, activation='softmax'))


# ### Solution

# Click on the '...' below to show the solution.

# ```python
# model = Sequential()
# model.add(Dense(units = 512, activation='relu', input_shape=(784,)))
# model.add(Dense(units = 512, activation='relu'))
# model.add(Dense(units = num_classes, activation='softmax'))
# ```

# ## Summarizing the Model

# Run the cell below to summarize the model you just created:

# In[20]:


model.summary()


# ## Compiling the Model

# We'll compile our model with the same options as before, using [categorical crossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy) to reflect the fact that we want to fit into one of many categories, and measuring the accuracy of our model:

# In[21]:


model.compile(loss='categorical_crossentropy', metrics=['accuracy'])


# ## Exercise: Train the Model

# Use your model's `fit` method to train it for 20 epochs using the training and testing images and labels you've created:

# In[22]:


history = model.fit(x_train, y_train,
                    epochs=20,
                    verbose=1,
                    validation_data=(x_test, y_test))


# ### Solution

# Click on the '...' below to show the solution.

# ```python
# model.fit(x_train, y_train,
#                     epochs=20,
#                     verbose=1,
#                     validation_data=(x_test, y_test))
# ```

# ## Discussion: What happened?

# We can see that the training accuracy got to a fairly high level, but the validation accuracy was not as high. What do you think happened here?
# 
# Think about it for a bit before clicking on the '...' below to reveal the answer.

# This is an example of the model learning to categorize the training data, but performing poorly against new data that it has not been trained on. Essentially, it is memorizing the dataset, but not gaining a robust and general understanding of the problem. This is a common issue called *overfitting*. We will discuss overfitting in the next two lectures, as well as some ways to address it.

# ## Summary

# In this section you built your own neural network to perform image classification, and while there is room for improvement, which we will get to shortly, you did quite well. Congrats!
# 
# At this point you should be getting somewhat familiar with the process of loading data (incuding labels), preparing it, creating a model, and then training the model with your prepared data.

# ### Clear the Memory
# Before moving on, please execute the following cell to clear up the GPU memory. This is required to move on to the next notebook.

# In[23]:


import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)


# ## Next

# Now that you have built some very basic, somewhat effective models, you will begin to learn about more sophisticated models, including *Convolutional Neural Networks*.
# 
# Please continue to the next section: [*ASL with CNNs*](./03_asl_cnn.ipynb).
