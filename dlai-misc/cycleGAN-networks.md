
THINGS TO UNDERSTAND
=====================

- networks

	ResNetGenerator (fast-neural-style)

	UNetGenerator    (UNet)
	UnetSkipConnectionBlock 

	NLayerDiscriminator (PatchGAN)
	
	PixelDiscriminator  (PatchGAN?	)

- losses
	GANLoss


REFERENCES
==========

- GAN networks
	https://spark-in.me/post/unet-adventures-part-one-getting-acquainted-with-unet

- fast-neural-style : 
	buscar explicacion/descrición de la red
	https://arxiv.org/abs/1508.06576
	https://arxiv.org/abs/1607.08022
	probar de reentrenarla


- UNet
	https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

	https://arxiv.org/pdf/1505.04597.pdf

	https://spark-in.me/post/unet-adventures-part-one-getting-acquainted-with-unet

	https://spark-in.me/post/gan-paper-review

	https://gluon.mxnet.io/chapter14_generative-adversarial-networks/pixel2pixel.html


- PatchGAN
	 Imageto-image translation with conditional adversarial networks.
	https://arxiv.org/pdf/1611.07004.pdf


- PixelDiscriminator
	Imageto-image translation with conditional adversarial networks.
	https://arxiv.org/pdf/1611.07004.pdf

	Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks
	https://arxiv.org/pdf/1604.04382.pdf

	Photo-realistic single image superresolution using a generative adversarial network
	https://arxiv.org/pdf/1609.04802.pdf


TASKS
======



- retrain with freezing 
	-> finetuning adapted
	-> for each network what layers to retrain?


- resize images 
	128x128
	64x64
	28x28


- download a pretrained model and retrain
	how to retrain download