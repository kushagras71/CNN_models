Model for Fashion_MNIST Dataset using KERAS API --- Accuracy 99.50%    
===============================================================
In the Fashion_MNIST notebook a model(classifier) is built using Convolutional Neural Network architecture.
After training on the Fashion MNIST, the model achieved an accuracy of 99.50%.
The architecture of the classifier is pretty simple and straight forward i.e. a direct application of CNN using Keras API. 

List of classes for the index value in the dataset :

0: T-shirt/top

1: Trouser

2: Pullover

3: Dress

4: Coat

5: Sandal

6: Shirt

7: Sneaker

8: Bag

9: Ankle boot

The summary of the model is as follows:

Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 25, 25, 64)        8256      
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 25, 25, 64)        4160      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 12, 12, 64)        0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 9216)              0         
_________________________________________________________________
dense_4 (Dense)              (None, 128)               1179776   
_________________________________________________________________
dense_5 (Dense)              (None, 10)                1290      
_________________________________________________________________
_________________________________________________________________
Total params: 1,193,802
Trainable params: 1,193,802
Non-trainable params: 0
_________________________________________________________________
