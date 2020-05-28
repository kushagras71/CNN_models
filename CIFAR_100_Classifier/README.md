CIFAR 100 Classifier -- 50% Accuracy
=
CIFAR 100 is a built in dataset found in the 'datasets' libraries  of Keras. This is datasets consists of 60,000 total images belonging to 100 different classes i.e.  600 images per class. For every class there are 500 training images and 100 test images. In total 50,000 training images and 10,000 test images.
The classes of the dataset are grouped into 20 super classes .Each image comes with a 'fine' label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).
Here is the list of classes in the CIFAR-100:


Superclass	Classes

<b>Aquatic mammals:</b> beaver, dolphin, otter, seal, whale

<b>Fish:</b>	aquarium fish, flatfish, ray, shark, trout

<b>Flowers:</b>	orchids, poppies, roses, sunflowers, tulips

<b>Food containers:</b>	bottles, bowls, cans, cups, plates

<b>Fruit and vegetables:</b>	apples, mushrooms, oranges, pears, sweet peppers

<b>Household electrical devices:</b>	clock, computer keyboard, lamp, telephone, television

<b>Household furniture:</b>	bed, chair, couch, table, wardrobe

<b>Insects:</b>	bee, beetle, butterfly, caterpillar, cockroach

<b>Large carnivores:</b>	bear, leopard, lion, tiger, wolf

<b>Large man-made outdoor things:</b>	bridge, castle, house, road, skyscraper

<b>Large natural outdoor scenes:</b>	cloud, forest, mountain, plain, sea

<b>Large omnivores and herbivores:</b>	camel, cattle, chimpanzee, elephant, kangaroo

<b>Medium-sized mammals:</b>	fox, porcupine, possum, raccoon, skunk

<b>Non-insect invertebrates:</b>	crab, lobster, snail, spider, worm

<b>People:</b>	baby, boy, girl, man, woman

<b>Reptiles:</b>	crocodile, dinosaur, lizard, snake, turtl

<b>Small mammals:</b>	hamster, mouse, rabbit, shrew, squirrel

<b>Trees:</b>	maple, oak, palm, pine, willow

<b>Vehicles 1:</b>	bicycle, bus, motorcycle, pickup truck, train

<b>Vehicles 2:</b>	lawn-mower, rocket, streetcar, tank, tractor

Reference <https://www.cs.toronto.edu/~kriz/cifar.html> 

 
The Python Notebook attached shows the architecture and the training processes of the CNN model. The model has around 338,000 parameters in total. Having many parameters, it was trained on a GPU. It is always helpful to run CNN and RNN models on GPUs because they require a lot of computational power. In this case Google Colab was used since it provides strong computational resources ( GPU and TPU support)  for free (a cloud based environment). A local GPU can also be used but setting up a GPU requires some research and version compatibility. There many useful videos on YouTube for the same. And before starting please make sure you have GPU device install on your system.  

After training the model achieved an accuracy of 50.95% on the training set and 45.69% on the validation (test) set. This accuracy is  pretty good for a CNN model not having an extremely deep network, but used classifying 60,000 images into 100 different classes. The accuracy can be increased by increasing the number of the images or using data augmentation.
