
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

Architecture
============

Generative networks
	Johnson et al. [23]
		- Perceptual losses for real-time style transfer and super-resolution.

	2 stride-2 conv
	residual blocks [18]
		- Deep residual learning for image recognition.

	2 fractionally-strided conv
	instance normalization [53]
		- Instance normalization: The missing ingredient for fast stylization.


Discriminator networks
	70x70 PatchGANs  [22,30,29]
		- Image-to-image translation with conditional adversarial networks.

		- Precomputed real-time texture synthesis with markovian generative adversarial networks.

		- Photo-realistic single image superresolution using a generative adversarial network.


REFERENCES
==========

- GAN networks
	https://spark-in.me/post/unet-adventures-part-one-getting-acquainted-with-unet

- fast-neural-style : 
	A neural algorithm of artistic Style
		https://arxiv.org/abs/1508.06576 
	Instance Normalization: The missing ingredient for fast stylization
		https://arxiv.org/abs/1607.08022 

	

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

- summaryze each network architecture

- retrain with freezing 
	-> finetuning adapted
	-> for each network what layers to retrain?


- resize images 
	128x128
	64x64
	28x28


- download a pretrained model and retrain
	how to retrain download


Summaries
==========


## Perceptual losses for real-time style transfer and super-resolution.

- from feed-forward CNN +  per-pixel loss 
   to 
   "perceptual" losses
   		differences between high-level image feature representations extracted from pretrained CNN


- inputx  -> image transf. network -> ^y -> loss network (VGG-16) -> 
                                      ys style target
                                      yc  content target

- perceptual loss is computed by a pretarined CNN -> the loss network by comparing ^y to ys and yc (style and content)


- image transformation networks
	no pooling layers
	strided and fractionally strided conv for downsampling and upsampling

	body
		1)5 residual blocks
				3x3 conv layers
				
		2) non-residual conv. layers: 
			non-residual conv layer
			spatial batch normalization
			ReLU (except for output whithc is scaled tanh)