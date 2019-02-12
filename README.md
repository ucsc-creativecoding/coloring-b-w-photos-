# coloring-b-w-photos
We followed the assignment of "transform images using an autoencoder neural network. Train an autoencoder DLNN that learns to emulate some form of image processing, such as colorizing black and white photos, or performing super resolution, etc".

We used 3000 pairs of images of flowers as training data and tested it on flower photos as well as other black and white photos. 

Here are example images from the training dataset:

![](Results/flower_training_coloring-04.jpg)


We tested the network with new black and white images that were not part of the original dataset. The results turn out pretty good, flowers were colored in a way that seems "right" or as you'd expect flowers would be colored.

![](Results/flower_training_coloring-02.jpg)


We then wanted to see how it would color images that have less attributes with the original dataset (e.g. not flowers..). We tested it with a random selection of black and white images. 
The result of the coloring of these images is more whimsical, interseting in and of itself. 

![](Results/flower_training_coloring-01.jpg)


![](Results/flower_training_coloring-03.jpg)
