#!/usr/bin/env python
# coding: utf-8

# <img src="./images/DLI_Header.png" style="width: 400px;">

# # Assessment

# Congratulations on going through today's course! Hopefully you've learned some valuable skills along the way. Now it's time to put those skills to the test. In this assessment you will train a new model that is able to recognize fresh and rotten fruit. You will need to get the model to a validation accuracy of 95% in order to pass the assessment. You will have the use the skills that you learned in the previous exercises. Specifically we suggest you use some combination of transfer learning, data augmentation, and fine tuning. Once you have trained the model to be at least 95% accurate on the test dataset, you will save your model, and then assess its accuracy. Let's get started! 

# ## The Dataset

# In this exercise, you will train a model to recognize fresh and rotten fruits. The dataset comes from [Kaggle](https://www.kaggle.com/sriramr/fruits-fresh-and-rotten-for-classification), a great place to go if you're interested in starting a project after this class. If you'd like, you can look at the dataset structure in the `fruits` folder. There are 6 categories of fruits: fresh apples, fresh oranges, fresh bananas, rotten apples, rotten oranges, and rotten bananas. This will mean that your model will require an output layer of 6 neurons to do the categorization successfully. You'll also need to compile the model with `categorical_crossentropy`, as we have more than two categories.

# <img src="./images/fruits.png" style="width: 600px;">

# ## Load ImageNet Base Model

# We encourage you to start with a model pretrained on ImageNet. You'll need to load the model with the correct weights, set an input shape, and choose to remove the last layers of the model. Remember that images have three dimensions: a height, and width, and a number of channels. Because these pictures are in color, there will be three channels for red, green, and blue. We've filled in the input shape for you. This cannot be changed or the assessment will fail. If you need a reference for setting up the pretrained model, please take a look at notebook 05b where you first implemented transfer learning.

# In[34]:


from tensorflow import keras

base_model = keras.applications.VGG16(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False)


# ## Freeze Base Model

# Next, we suggest you freeze the base model, as you did in notebook 05b. This is done so that all the learning from the ImageNet dataset does not get destroyed in the initial training.

# In[35]:


# Freeze base model
base_model.trainable = False


# ## Add Layers to Model

# Now it's time to add layers to your pretrained model. You can again use notebook 05b as a guide. Pay close attention to the last dense layer and make sure it has the correct number of neurons to classify the different types of fruit.

# In[36]:


# Create inputs with correct shape
inputs = keras.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)

# Add pooling layer or flatten layer
x = keras.layers.GlobalAveragePooling2D()(x)

# Add final dense layer
outputs = keras.layers.Dense(6, activation = 'softmax')(x)

# Combine inputs and outputs to create model
model = keras.Model(inputs, outputs)


# In[37]:


model.summary()


# ## Compile Model

# Now it's time to compile the model with loss and metrics options. Remember that we're training on a number of different categories, rather than a binary classification problem.

# In[38]:


model.compile(loss='categorical_crossentropy', metrics=['accuracy'])


# ## Augment the Data

# If you'd like, try to augment the data to improve the dataset. Feel free to look at notebook 04a and notebook 05b for augmentation examples. You can also look at the documentation for the [Keras ImageDataGenerator class](https://keras.io/api/preprocessing/image/#imagedatagenerator-class). This step is optional, but you may find it helpful to get to 95% accuracy when you train.

# In[39]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False) # we don't expect the fruit to be upside-down so we will not flip vertically


# ## Load Dataset

# Now it's time to load the train and test datasets. You'll have to pick the right folders, as well as the right `target_size` of the images (it needs to match the height and width input of the model you've created). If you'd like a reference, you can check out notebook 05b.

# In[40]:


# load and iterate training dataset
train_it = datagen.flow_from_directory('fruits/train/', 
                                       target_size=(224,224),
                                       color_mode='rgb', 
                                       class_mode="categorical")
# load and iterate test dataset
test_it = datagen.flow_from_directory('fruits/test/', 
                                      target_size=(224,224) ,
                                      color_mode='rgb', 
                                      class_mode="categorical")


# ## Train the Model

# Time to train the model! Pass the `train` and `test` iterators into the `fit` function, as well as setting your desired number of epochs.

# In[41]:


model.fit(train_it,
          validation_data=test_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=test_it.samples/test_it.batch_size,
          epochs=20)


# ## Unfreeze Model for Fine Tuning

# If you have reached 95% validation accuracy already, this next step is optional. If not, we suggest you try fine tuning the model with a very low learning rate. You may again use notebook 05b as a reference.

# In[42]:


# Unfreeze the base model
base_model.trainable = True

# Compile the model with a low learning rate
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate = 0.00001),
              loss = keras.losses.CategoricalCrossentropy(from_logits=True) , 
              metrics = [keras.metrics.CategoricalAccuracy()])


# In[43]:


model.fit(train_it,
          validation_data=test_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=test_it.samples/test_it.batch_size,
          epochs=20)


# ## Evaluate the Model

# Hopefully you now have a model that has a validation accuracy of 95% or higher. If not, you may want to go back and either run more epochs of training, or adjust your data augmentation. 
# 
# Once you are satisfied with the validation accuracy, you can evaluate the model by executing the following cell. The evaluate function will return a tuple, where the first value is your loss, and the second value is your accuracy. You'll want to have an accuracy value of 95% or higher. 

# In[44]:


model.evaluate(test_it, steps=test_it.samples/test_it.batch_size)


# ## Save the Model

# Now it's time to save the model. The assessment grader will look for this model and verify that you have above 95% accuracy.

# In[45]:


model.save('assessment_model')


# ### Clear the Memory
# Before assessing the model, you must run the following cell to clear up the GPU memory.

# In[46]:


import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)


# ## Assess the Model

# In order to assess the model, please return to the course page and click the assess button. You will be notified if you've passed, and will receive a certificate upon successful completion.

# <img src="./images/assess_task.png" style="width: 800px;">
