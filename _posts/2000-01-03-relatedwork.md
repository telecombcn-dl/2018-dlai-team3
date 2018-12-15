---
title: "Related Work"
bg: #9AD1F5
color: black
style: center
fa-icon: puzzle-piece
---

When it comes to image to image translation there have been several implementations using neural networks that preceed cycleGANs.

### [A neural algorithm for artistic style](https://arxiv.org/abs/1508.06576)

![Proposed network](./img/basicstyletransfer.png)

The first implementation of style transfer in the realm of neural networks took a CNN trained for image classification and separated the part of it that learnt texture information. Applying it to a new image. It used a square error loss  to minimize the distance between the 2 feature representations extracted from passing a white noise image and a picture trough the network.

### [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)

In this paper the concept of Generative adversarial networks is introduced. It sets a generator network that generates images from noise and a discriminator network that tells wether it's input is a real image and trains them both against one another.


