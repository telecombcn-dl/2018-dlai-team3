---
title: "The puzzle pieces"
bg: #9AD1F5
color: black
style: center
fa-icon: puzzle-piece
---

# A brief history of image to image solutions

### [A neural algorithm for artistic style](https://arxiv.org/abs/1508.06576)

![Proposed network](./img/basicstyletransfer.png)

The first implementation of style transfer in the realm of neural networks took a CNN trained for image classification and separated the part of it that learnt texture information. Applying it to a new image. It used a square error loss  to minimize the distance between the 2 feature representations extracted from passing a white noise image and a picture trough the network.

### [*ResNet* Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

![Residual learning, a building blocl](./img/residual.png)

In this paper the concept of residual learning is introduced. While not a specific image to image concept it's a building block of most of the implementations we studied.
It works by explicitly fitting to the residue of the mapping. This makes the network easier to optimize and thus makes deeper networks viable. Which in turn increases accuracy.

### [*Instance Normalization*:The Missing Ingredient for Fast Stylization](https://arxiv.org/pdf/1607.08022.pdf)

This paper shows how changing from batch normalization to instance normalization improves performance for image generators. Instead of normalizing for the whole batch of images it normalizes each instance. 
> This prevents instance-specific mean and covariance shift simplifying the learning process.

### [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)

In this paper the concept of Generative adversarial networks is introduced. It sets a generator network that generates images from noise and a discriminator network that tells wether it's input is a real image and trains them both against one another.
This will be further explaned later on. 

### [Photo-Realistic Single Image Super-Resolution Using a Generative AdversarialNetwork](https://arxiv.org/pdf/1609.04802.pdf)

