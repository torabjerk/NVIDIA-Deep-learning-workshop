#!/usr/bin/env python
# coding: utf-8

# <img src="./images/DLI_Header.png" style="width: 400px;">

# # Transfer Learning

# So far we've trained accurate models on large datasets, and also downloaded a pre-trained model that we used with no training necessary. But what if you can't find a pre-trained model that does exactly what you need, and what if you don't have a sufficiently large dataset to train a model from scratch? In this case, there's a very helpful technique we can use called [transfer learning](https://blogs.nvidia.com/blog/2019/02/07/what-is-transfer-learning/).
# 
# With transfer learning, you take a pre-trained model and retrain it on a task that has some overlap with the original training task. A good analogy for this is an artist who's skilled in one medium, such as painting, who wants to learn to practice in another medium, such as charcoal drawing. You can imagine that the skills they learned while painting would be very valuable in learning how to draw with charcoal. 
# 
# As an example in deep learning, say you have a pre-trained model that is very good at recognizing different types of cars, and you want to train a model to recognize types of motorcycles. A lot of the learnings of the car model would likely be very useful, for instance the ability to recognize headlights and wheels. 
# 
# Transfer learning is especially powerful when you do not have a large and varied dataset. In this case, a model trained from scratch would likely memorize the training data quickly, but not be able to generalize well to new data. With transfer learning, you can increase your chances of training an accurate and robust model on a small dataset.

# ## Objectives

# By the time you complete this section you will be able to:
# * Prepare a pretrained model for transfer learning
# * Perform transfer learning with your own small dataset on a pretrained model
# * Further fine tune the model for even better performance

# ## A Personalized Doggy Door

# In our last exercise, we used a pre-trained [ImageNet](http://www.image-net.org/) model to let in all dogs, but keep out other animals. In this exercise, we'd like to create a doggy door that only lets in a particular dog. In this case, we'll make an automatic doggy door for a dog named Bo. You might recognize Bo as President Barack Obama's family's dog. You can find pictures of Bo in the `presidential_doggy_door` folder.

# <img src="presidential_doggy_door/train/bo/bo_10.jpg">

# The challenge is that the pre-trained model was not trained to recognize this specific dog, and, we only have 30 pictures of Bo. If we tried to train a model from scratch using those 30 pictures we would experience overfitting and poor generalization. However, if we start with a pre-trained model that is adept at detecting dogs, we can leverage that learning to gain a generalized understanding of Bo using our smaller dataset. We can use transfer learning to solve this challenge.

# ## Downloading the Pretrained Model

# The ImageNet pre-trained models are often good choices for computer vision transfer learning, as they have learned to classify many many different types of images. In doing this, they have learned to detect many different types of [features](https://developers.google.com/machine-learning/glossary#) that could be valuable in image recognition. Because ImageNet models have learned to detect animals, including dogs, it's especially well suited for this transfer learning task of detecting Bo.
# 
# We'll start by downloading the pre-trained model. Again, this is available directly from the Keras library. As we're downloading, there's going to be an important difference. The last layer of an ImageNet model is a [dense layer](https://developers.google.com/machine-learning/glossary#dense-layer) of 1000 units, representing the 1000 possible classes in the dataset. In our case, we want it to make a different classification: is this Bo or not? Because we want the classification to be different, we are going to remove the last layer of the model. We can do this by setting the flag `include_top=False` when downloading the model. After removing this top layer, we can add new layers that will yield the type of classification that we want:

# In[1]:


from tensorflow import keras

base_model = keras.applications.VGG16(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(224, 224, 3),
    include_top=False)


# In[2]:


base_model.summary()


# ## Freezing the Base Model
# Before we add our new layers onto the [pre-trained model](https://developers.google.com/machine-learning/glossary#pre-trained-model), we should take an important step: freezing the model's pre-trained layers. This means that when we train, we will not update the base layers from the pre-trained model. Instead we will only update the new layers that we add on the end for our new classification. We freeze the initial layers because we want to retain the learning achieved from training on the ImageNet dataset. If they were unfrozen at this stage, we would likely destroy this valuable information. There will be an option to unfreeze and train these layers later, in a process called fine-tuning.
# 
# Freezing the base layers is as simple as setting trainable on the model to false.

# In[3]:


base_model.trainable = False


# ## Adding New Layers

# We can now add the new trainable layers on top of the pre-trained model. They will take the features from the pre-trained layers and turn them into predictions on the new dataset. We'll add two layers to the model. First will be a pooling layer like we saw in our earlier [convolutional neural network](https://developers.google.com/machine-learning/glossary#convolutional_layer). (If you want a more thourough understanding of the role of pooling layers in CNNs, please read [this detailed blog post](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/#:~:text=A%20pooling%20layer%20is%20a,Convolutional%20Layer)). We then need to add our final layer, which will classify Bo or not Bo. This will be a densely connected layer with one output.

# In[5]:


inputs = keras.Input(shape=(224, 224, 3))
# Separately from setting trainable on the model, we set training to ase 
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)


# Let's take a look at the model, now that we've combined the pre-trained model with the new layers.

# In[6]:


model.summary()


# Keras gives us a nice summary here, as it shows the vgg16 pre-trained model as one unit, rather than showing all of the internal layers. It's also worth noting that we have many non-trainable parameters as we've frozen the pre-trained model. 

# ## Compiling the Model

# As with our previous exercises, we need to compile the model with loss and metrics options. We have to make some different choices here. In previous cases we had many categories in our classification problem. As a result, we picked categorical crossentropy for the calculation of our loss. In this case we only have a binary classification problem (Bo or not Bo), and so we'll use [binary crossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy). If you'd like to learn more about the differences between the two you can read more [here](https://gombru.github.io/2018/05/23/cross_entropy_loss/). We will also use binary accuracy instead of traditional accuracy.
# 
# By setting `from_logits=True` we inform the [loss function](https://gombru.github.io/2018/05/23/cross_entropy_loss/) that the output values are not normalized (e.g. with softmax).

# In[7]:


# Important to use binary crossentropy and binary accuracy as we now have a binary classification problem
model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=[keras.metrics.BinaryAccuracy()])


# ## Augmenting the Data

# Now that we're dealing with a very small dataset, it's especially important that we augment our data. As before, we'll make small modifications to the existing images, which will allow the model to see a wider variety of images to learn from. This will help it learn to recognize new pictures of Bo instead of just memorizing the pictures it trains on.

# In[8]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
# create a data generator
datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False) # we don't expect Bo to be upside-down so we will not flip vertically


# ## Loading the Data

# We've seen datasets in a couple different formats so far. In the MNIST exercise, we were able to download the dataset directly from within the Keras library. For the sign language dataset, the data was in CSV files. For this exercise, we're going to load images directly from folders using Keras' [`flow_from_directory`](https://keras.io/api/preprocessing/image/) function. We've set up our directories to help this process go smoothly. In the `presidential_doggy_door` directory, we have train and test directories, which each have folders for images of Bo and not Bo. In the not_bo directories, we have pictures of other dogs and cats, to teach our model to keep out other pets. Feel free to explore the images to get a sense of our dataset.
# 
# Note that [flow_from_directory](https://keras.io/api/preprocessing/image/) will also allow us to size our images to match the model: 244x244 pixels with 3 channels. (Note to reviewers: I may want to add a bit about batch_size here, as we've not covered it yet)

# In[9]:


# load and iterate training dataset
train_it = datagen.flow_from_directory('presidential_doggy_door/train/', 
                                       target_size=(224, 224), 
                                       color_mode='rgb', 
                                       class_mode='binary', 
                                       batch_size=8)
# load and iterate test dataset
test_it = datagen.flow_from_directory('presidential_doggy_door/test/', 
                                      target_size=(224, 224), 
                                      color_mode='rgb', 
                                      class_mode='binary', 
                                      batch_size=8)


# ## Training the Model

# Now it's time to train our model and see how it does. Recall that when using a data generator, we have to explicitly set the number of `steps_per_epoch`:

# In[10]:


model.fit(train_it, steps_per_epoch=12, validation_data=test_it, validation_steps=4, epochs=20)


# ## Discussion of Results

# You'll hopefully find that both the training and validation accuracy are quite high. This is a pretty awesome result! We were able to train on a small dataset, but because of the knowledge transferred from the ImageNet model, it was able to achieve high accuracy and generalize well. This means it has a very good sense of Bo and pets who are not Bo.
# 
# If you saw some fluctuation in the validation accuracy, that's okay too. We have a technique for improving our model in the next section.

# ## Fine-Tuning the Model

# Now that the new layers of the model are trained, we have the option to apply a final trick to improve the model, called [fine-tuning](https://developers.google.com/machine-learning/glossary#f). To do this we unfreeze the entire model, and train it again with a very small [learning rate](https://developers.google.com/machine-learning/glossary#learning-rate). This will cause the base pre-trained layers to take very small steps and adjust slightly, improving the model by a small amount.  
# 
# Note that it's important to only do this step after the model with frozen layers has been fully trained. The untrained pooling and classification layers that we added to the model earlier were randomly initialized. This means they needed to be updated quite a lot to correctly classify the images. Through the process of [backpropagation](https://developers.google.com/machine-learning/glossary#backpropagation), large initial updates in the last layers would have caused potentially large updates in the pre-trained layers as well. These updates would have destroyed those important pre-trained features. However, now that those final layers are trained and have converged, any updates to the model as a whole will be much smaller (especially with a very small learning rate) and will not destroy the features of the earlier layers.
# 
# Let's try unfreezing the pre-trained layers, and then fine tuning the model:

# In[11]:


# Unfreeze the base model
base_model.trainable = True

# It's important to recompile your model after you make any changes
# to the `trainable` attribute of any inner layer, so that your changes
# are taken into account
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate = .00001),  # Very low learning rate
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])


# In[12]:


model.fit(train_it, steps_per_epoch=12, validation_data=test_it, validation_steps=4, epochs=10)


# ## Examining the Predictions

# Now that we have a well-trained model, it's time to create our doggy door for Bo! Let's start by looking at the predictions that come from the model. We'll preprocess the image in the same way we did for our last doggy door.

# In[13]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input

def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image)

def make_predictions(image_path):
    show_image(image_path)
    image = image_utils.load_img(image_path, target_size=(224, 224))
    image = image_utils.img_to_array(image)
    image = image.reshape(1,224,224,3)
    image = preprocess_input(image)
    preds = model.predict(image)
    return preds


# Let's try this out on a couple images to see the predictions:

# In[14]:


make_predictions('presidential_doggy_door/test/bo/bo_20.jpg')


# In[14]:


make_predictions('presidential_doggy_door/test/not_bo/121.jpg')


# It looks like a negative number prediction means that it's Bo and a positive number prediction means it's something else. We can use this information to have our doggy door only let Bo in! 

# ## Exercise: Bo's Doggy Door

# Fill in the following code to implement Bo's doggy door:

# In[16]:


def presidential_doggy_door(image_path):
    preds = make_predictions(image_path)
    if preds<0:
        print("It's Bo! Let him in!")
    else:
        print("That's not Bo! Stay out!")


# ## Solution

# Click on the '...' below to see the solution.

# ```python
# def presidential_doggy_door(image_path):
#     preds = make_predictions(image_path)
#     if preds[0] < 0:
#         print("It's Bo! Let him in!")
#     else:
#         print("That's not Bo! Stay out!")
# ```

# Let's try it out!

# In[17]:


presidential_doggy_door('presidential_doggy_door/test/not_bo/131.jpg')


# In[18]:


presidential_doggy_door('presidential_doggy_door/test/bo/bo_29.jpg')


# ## Summary

# Great work! With transfer learning, you've built a highly accurate model using a very small dataset. This can be an extremely powerful technique, and be the difference between a successful project and one that can't get off the ground. We hope these techniques can help you out in similar situations in the future!
# 
# You can get a wealth of helpful resources for transfer learning from the [NVIDIA Transfer Learning Toolkit](https://developer.nvidia.com/tlt-getting-started).

# ### Clear the Memory
# Before moving on, please execute the following cell to clear up the GPU memory.

# In[1]:


import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)


# ## Next

# So far the focus of this workshop has primarily been on image classification. In the next section, in service of giving you a more well-rounded introduction to deep learning, we are going to switch gears and address working with sequential data, which requires a different approach.
# 
# Please continue to the next section: [*Sequence Data*](./06_headline_generator.ipynb).
