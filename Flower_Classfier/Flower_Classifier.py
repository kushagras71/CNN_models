import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread

# =============================================================================
# Assigning the data set dictory path  ( Depends on your system )
# =============================================================================
data_dir = '.\flower_photos'
os.listdir(data_dir)

# =============================================================================
# Assigning the train and test set directory path
# =============================================================================
train_set = data_dir + '\\train\\'
test_set  = data_dir + '\\test\\'

# =============================================================================
#  Displaying a sample image from the training set
# =============================================================================
os.listdir(train_set + '\\daisy\\')[0]
sample_image = train_set + '\\daisy\\'+'100080576_f52e8ee070_n.jpg'
plt.imshow(imread(sample_image))

# =============================================================================
#  Assigning a common dimension for all the images for traning 
# =============================================================================

image_shape = (28,28,3)

# =============================================================================
# Creating Image Data Generator instance for data agumentation
# =============================================================================
from tensorflow.keras.preprocessing.image import ImageDataGenerator
img_gen = ImageDataGenerator(rotation_range=20,
                                           width_shift_range = 0.1,
                                           height_shift_range=0.1,
                                           shear_range=0.1,
                                           zoom_range=0.1,
                                           horizontal_flip = True,
                                           fill_mode = 'nearest')


#showing the sample working of image data generator
sample_im.shape = imread(sample_image)
plt.imshow(sample_im)

# show a sample image of a randomly augumented image 
plt.imshow(img_gen.random_transform(sample_im))

# the below tow code lines displays the labeling of the train and test
img_gen.flow_from_directory(train_set)
img_gen.flow_from_directory(test_set)


# =============================================================================
# Creating the data augumentoin instance for the train and test set
# =============================================================================
batch_size = 16
train_image_gen  = img_gen.flow_from_directory(train_set,
                                               target_size = image_shape[:2],
                                               color_mode = 'rgb',
                                               batch_size = batch_size,
                                               class_mode = 'binary')
test_image_gen = img_gen.flow_from_directory(test_set,
                                             target_size = image_shape[:2],
                                             color_mode = 'rgb',
                                             batch_size = batch_size,
                                             class_mode = 'binary')

train_image_gen.class_indices
test_image_gen.class_indices

# output of the above two line is same as follows:
    '''
    {'daisy': 0, 'dandelion': 1, 'roses': 2,
     'sunflowers': 3, 'tulips': 4}
    '''


# =============================================================================
# Creating the model
# =============================================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,MaxPooling2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
model = Sequential()

model.add(Conv2D(filters=128,kernel_size=(3,3),input_shape = image_shape,activation='relu',padding='valid'))
model.add(Conv2D(filters=64,kernel_size=(2,2),input_shape = image_shape,activation='relu',padding='valid'))
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape = image_shape,activation='relu',padding='valid'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape = image_shape,activation='relu',padding='valid'))
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape = image_shape,activation='relu',padding='valid'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=32,activation='relu'))
model.add(Dense(units=5,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Instance of early stopping
early_stop = EarlyStopping(monitor='val_loss',patience=10)

#Fitting the model for the training set and validating on the test
model.fit_generator(train_image_gen,epochs=50,validation_data=(test_image_gen),callbacks=[early_stop])


# =============================================================================
# Storing the results after training the model
# =============================================================================
results = pd.DataFrame(model.history.history)
results.plot()
results[['loss','val_loss']].plot()
results[['accuracy','val_accuracy']].plot()


# =============================================================================
# Saving the model on the system in the working directory
# =============================================================================
model.save('Flower_Classifier_h5')


# =============================================================================
# Evaluting the model on the test set
# =============================================================================
print('Loss: ',model.evaluate_generator(test_image_gen)[0],
      'Accuracy: ',model.evaluate_generator(test_image_gen)[1])

# =============================================================================
# Predicting the a new image 
# You can either choose from the test set or use new image
# =============================================================================


test_img = (# Enter the correct path for the new image you are using for the predict )
            # In my case a random image of the daisy flower from test set was picked.
            # Note: We know that image is of a daisy flower but the model doesn't know.
            # Lets check it. 
plt.imshow(imread(test_img))  # Displaying the new image

from tensorflow.keras.preprocessing import image


new_img = image.load_img(test_img,target_size=image_shape) #loading image as a tensor for the model
new_img   # Checking the image after converting it to tensor
new_img_array = image.img_to_array(new_img)  # converting the tensor to an array form on which the model predict on

new_img_array.shape  #(28,28,3)
new_img_array = np.expand_dims(new_img_array,axis=0)  # (1,28,28,3) for batch size

probabilities = model.predict(new_img_array)  #predicting the probabilites of the classes

print(list(train_image_gen.class_indices)[0],":",probabilities[0][0],'\n',
      list(train_image_gen.class_indices)[1],":",probabilities[0][1],'\n',
      list(train_image_gen.class_indices)[2],":",probabilities[0][2],'\n',
      list(train_image_gen.class_indices)[3],":",probabilities[0][3])


##### The model predicts the highest probabilty for the class 'daisy'###########
########## Please check the Read me file for the snap shot of the probabilites## 











