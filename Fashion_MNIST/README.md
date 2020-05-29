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

The architecture of the model is pretty straight forward but it has more than 1 million parameters so it is adviced to run these CNN architectures on GPUs since CNNs require a lot of computational power which CPUs are not capable off. Even if you run CNN mdoels on CPU it would take quite a long time to train and converge more the results may not be that good in comparison to GPU which will offer low training time and high accuracy.

The model in this notebook was trained on Google Colab a cloud based programming eviornment with GPU and TPU support.
