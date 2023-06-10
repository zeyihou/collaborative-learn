# collaborative-learn
This is the official implementation of "WL-GAN: Learning where to sample in generative latent space", which boost the sample quality of trained GANs with collaborative learning in latent space sampling.

The framework is inherited from [REPGAN](https://github.com/yifeiwang77/repgan), thanks for their works and sharing..

## Abstract
Recent advances in generative latent space sampling for enhanced generation quality have demonstrated the benefits from the Energy-Based Model (EBM), which is often defined by both the generator and the discriminator of off-the-shelf Generative Adversarial Networks (GANs) of many types. However, such latent space sampling may still suffer from mode dropping even sampling in a low-dimensional latent space, due to the inherent complexity of the data distributions with rugged energy landscapes. In this paper, we propose WL-GAN, a collaborative learning framework for generative latent space sampling, where both the invariant distribution and the proposal distribution are jointly learned on the fly, by exploiting the historical statistics behind the samples of the Markov chain. We show that the two learning modules work together for better balance between exploration and exploitation over the energy space in GAN sampling, alleviating mode dropping and improving the sample quality of GAN. Empirically, the efficacy of WL-GAN is demonstrated on both synthetic datasets and real-world image datasets, using multiple GANs.

## Requirements
`torch`
`torchvision`
`numpy`
`tqdm`


## Data
We investigate the collaborative learning on three real-world image datasets, including ```CIFAR-10```, ```CelebA``` and ```ImageNet-100``` (a 100-class subset of ImageNet), with different GAN models.

## GAN models
We train three different GANs (i.e., ```DCGAN```, ```WGAN```, and ```SNGAN```) on CIFAR-10 and CelebA for empirical evaluation. As for ImageNet-100, we only implement ```BigGAN``` because the training of other baseline models is unstable.

Notice: We take ```DCGAN```

## Usage

## Acknowledgment
Our project references the codes in the following repos. Thanks for their works and sharing.
- [REPGAN](https://github.com/yifeiwang77/repgan)
- [MHGAN](https://github.com/uber-research/metropolis-hastings-gans)
- [DCGAN](https://github.com/pytorch/examples/blob/master/dcgan/main.py)
- [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch)
