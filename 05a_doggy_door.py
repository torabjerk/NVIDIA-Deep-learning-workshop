#!/usr/bin/env python
# coding: utf-8

# <img src="./images/DLI_Header.png" style="width: 400px;">

# # Pre-Trained Models

# Though it's often necessary to have a large, well annotated dataset to solve a deep learning challenge, there are many  freely available pre-trained models that we can use right out of the box. As you decide to take on your own deep learning project, it's a great idea to start by looking for existing models online that can help you achieve your goal. A great place to explore available models is [Model Zoo](https://modelzoo.co/). There are also many models hosted on Github that you can find through searching on Google. 

# ## Objectives

# By the time you complete this section you will be able to:
# * Use Keras to load a very well-trained pretrained model
# * Preprocess your own images to work with the pretrained model
# * Use the pretrained model to perform accurate inference on your own images

# ## An Automated Doggy Door

# In this section, we'll be creating a doggy door that only lets dogs (and not other animals) in and out. We can keep our cats inside, and other animals outside where they belong. Using the techniques covered so far, we'd need a very large dataset with pictures of many dogs, as well as other animals. Luckily there's a readily available model that's been trained on a massive dataset, including lots of animals. 
# 
# The [ImageNet challenge](https://en.wikipedia.org/wiki/ImageNet#History_of_the_ImageNet_challenge) has produced many state-of-the-art models that can be used for image classification. They are trained on millions of images, and can accurately classify images into 1000 different categories. Many of those categories are animals, including breeds of dogs and cats. This is a perfect model for our doggy door.

# ## Loading the Model

# We'll start by downloading the model. Trained ImageNet models are available to download directly within the Keras library. You can see the available models and their details [here](https://keras.io/api/applications/#available-models). Any of these models would work for our exercise. We'll pick a commonly used one called [VGG16](https://keras.io/api/applications/vgg/):

# In[1]:


from tensorflow.keras.applications import VGG16
  
# load the VGG16 network *pre-trained* on the ImageNet dataset
model = VGG16(weights="imagenet")


# Now that it's loaded, let's take a look at the model. You'll notice that it looks a lot like our convolutional model from the sign language exercise. Pay attention to the first layer (the input layer) and the last layer (the output layer). As with our earlier exercises, we need to make sure our images match the input dimensions that the model expects. It's also valuable to understand what the model will return from the final output layer.

# In[2]:


model.summary()


# ### Input dimensions
# We can see that the model is expecting images in the shape of (224,224,3) corresponding to 224 pixels high, 224 pixels wide, and 3 color channels. As we learned in our last exercise, Keras models can accept more than one image at a time for prediction. If we pass in just one image, the shape will be (1,224,224,3). We will need to make sure that when passing images into our model for prediction, they match these dimensions. 
# 
# ### Output dimensions
# We can also see that the model will return a prediction of shape 1000. Remember that in our first exercise the output shape of our model was 10, corresponding to the 10 different digits. In our second exercise we had a shape of 24, corresponding to the 24 letters of the sign language alphabet that could be captured in a still image. Here, we have 1000 possible categories that the image will be placed in. Though the full ImageNet dataset has over 20,000 categories, the competition and resulting pre-trained models just use a subset of 1000 of these categories. We can take a look at all of these [possible categories here](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). 
# 
# Many of the categories are animals, including many types of dogs and cats. The dogs are categories 151 through 268. The cats are categories 281 through 285. We will be able to use these categories to tell our doggy door what type of animal is at our door, and whether we should let them in or not.

# ## Loading an Image
# We'll start by loading in an image and displaying it, as we've done in previous exercises.

# In[3]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(image_path):
    image = mpimg.imread(image_path)
    print(image.shape)
    plt.imshow(image)


# In[4]:


show_image("doggy_door_images/happy_dog.jpg")


# ## Preprocessing the Image

# Next, we need to preprocess the image so that it's ready to be sent into the model. This is just like what we did in our last exercise when we predicted on the sign language images. Remember that in this case, the final shape of the image needs to be (1,224,224,3).
# 
# When loading models directly with Keras, we can also take advantage of [`preprocess_input` methods](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/preprocess_input). These methods, associated with a specific model, allow users to preprocess their own images to match the qualities of the images that the model was originally trained on. You had to do this manually yourself when performing inference with new ASL images:

# In[5]:


from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.vgg16 import preprocess_input

def load_and_process_image(image_path):
    # Print image's original shape, for reference
    print('Original image shape: ', mpimg.imread(image_path).shape)
    
    # Load in the image with a target size of 224,224
    image = image_utils.load_img(image_path, target_size=(224, 224))
    # Convert the image from a PIL format to a numpy array
    image = image_utils.img_to_array(image)
    # Add a dimension for number of images, in our case 1
    image = image.reshape(1,224,224,3)
    # Preprocess image to align with original ImageNet dataset
    image = preprocess_input(image)
    # Print image's shape after processing
    print('Processed image shape: ', image.shape)
    return image


# In[6]:


processed_image = load_and_process_image("doggy_door_images/brown_bear.jpg")


# ## Make a Prediction

# Now that we have our image in the right format, we can pass it into our model and get a prediction. We're expecting an output of an array of 1000 elements, which is going to be difficult to read. Fortunately, models loaded directly with Keras have yet another helpful method that will translate that prediction array into a more readable form. 
# 
# Fill in the following function to implement the prediction:

# In[7]:


from tensorflow.keras.applications.vgg16 import decode_predictions

def readable_prediction(image_path):
    # Show image
    image = mpimg.imread(image_path)
    plt.imshow(image)
    # Load and pre-process image
    image = image_utils.load_img(image_path, target_size=(224, 224))
    image = image_utils.img_to_array(image)
    image = image.reshape(1,224,224,3)
    image = preprocess_input(image)
    # Make predictions
    predictions = model.predict(image)
    # Print predictions in readable form
    print('Predicted:', decode_predictions(predictions, top=3))


# ### Solution

# Click on the '...' below to see the solution.

# ```python
# from tensorflow.keras.applications.vgg16 import decode_predictions
# 
# def readable_prediction(image_path):
#     # Show image
#     show_image(image_path)
#     # Load and pre-process image
#     image = load_and_process_image(image_path)
#     # Make predictions
#     predictions = model.predict(image)
#     # Print predictions in readable form
#     print('Predicted:', decode_predictions(predictions, top=3))
# ```

# Try it out on a few animals to see the results! Also feel free to upload your own images and categorize them. You might be surprised at how good this model is.

# In[8]:


readable_prediction("doggy_door_images/happy_dog.jpg")


# In[9]:


readable_prediction("doggy_door_images/brown_bear.jpg")


# In[10]:


readable_prediction("doggy_door_images/sleepy_cat.jpg")


# ## Only Dogs

# Now that we're making predictions with our model, we can use our categories to only let dogs in and out and keep cats inside. Remember that dogs are categories 151 through 268 and cats are categories 281 through 285. You can use the [np.argmax](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html) function to find which element of the prediction array is the top category.

# In[18]:


import numpy as np

def doggy_door(image_path):
    show_image(image_path)
    image = load_and_process_image(image_path)
    preds = model.predict(image)
    if np.argmax(preds)>=151 and np.argmax(preds)<=268:
        print("Doggy come on in!")
    elif np.argmax(preds)>=281 and np.argmax(preds)<=285:
        print("Kitty stay inside!")
    else:
        print("You're not a dog! Stay outside!")


# ### Solution

# Click on the '...' below to see the solution.

# ```python
# import numpy as np
# 
# def doggy_door(image_path):
#     show_image(image_path)
#     image = load_and_process_image(image_path)
#     preds = model.predict(image)
#     if 151 <= np.argmax(preds) <= 268:
#         print("Doggy come on in!")
#     elif 281 <= np.argmax(preds) <= 285:
#         print("Kitty stay inside!")
#     else:
#         print("You're not a dog! Stay outside!")
# ```

# In[19]:


doggy_door("doggy_door_images/brown_bear.jpg")


# In[20]:


doggy_door("doggy_door_images/happy_dog.jpg")


# In[21]:


doggy_door("doggy_door_images/sleepy_cat.jpg")


# ## Summary

# Great work! Using a powerful pre-trained model, we've created a functional doggy door in just a few lines of code. We hope you're excited to realize that you can take advantage of deep learning without a lot of up-front work. The best part is, as the deep learning community moves forward, more models will become available for you to use on your own projects.

# ### Clear the Memory
# Before moving on, please execute the following cell to clear up the GPU memory.

# In[22]:


import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)


# ## Next

# Using pretrained models is incredibly powerful, but sometimes they are not a perfect fit for your data. In the next section you will learn about another powerful technique, *transfer learning*, which allows you to tailer pretrained models to make good predictions for your data.
# 
# Continue to the next section: [*Pretrained Models*](./05b_presidential_doggy_door.ipynb).
