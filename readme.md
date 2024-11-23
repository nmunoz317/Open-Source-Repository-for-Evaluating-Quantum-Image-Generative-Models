# Open-Source Repository for Evaluating Quantum Image Generative Models

**This repository focuses on benchmarking quantum generative models in the field of quantum machine learning, specifically for image generation. While previous projects have primarily addressed simple image distributions, our work aims to evaluate how these models perform across different image resolutions. We provide an open-source codebase that includes several quantum generative architectures. This repository is open-access, encouraging individuals to contribute and enhance our understanding of the current capabilities of these models.**

## Table of Contents
  * [Usage](#usage)
  * [Datasets](#datasets)
    * [Blood MedMNIST](#blood-medmnist)
    * [BraTS dataset](#brats-dataset)
  * [Quantum GANs](#quantum-gans)
    * [Quantum Patch GAN](#quantum-patch-gan)
    * [QC-GAN](#qc-gan)
    * [PQWGAN](#pqwgan)
  * [Classical GANs](#classical-gans)
    * [GAN](#gan)
    * [WGAN](#wgan)
  * [Quantum Denoising Diffusion Models](#quantum-denoising-diffusion-models)
    * [Q-Dense and Q-Dense Directed](#q-dense-and-q-dense-directed)
  * [Classical Denoising Diffusion Models](#classical-denoising-diffusion-models)
    * [Denoising Diffusion Probabilistic Architecture](#denoising-diffusion-probabilistic-architecture)



## Usage
Install the requirements: `pip install -r requirements.txt`.

- Run `python train.py --help` to see all the options.
- Run `python train.py --help+ dataset/model` to see the required arguments for different datasets and architectures.
- Run `python train.py` to train and evaluate the different models in the different datasets.

## Datasets

This project focuses on the use of biomedical images. To this end, two datasets have been selected: one that contains low-resolution images and another that contains more complex images with considerable resolution.


### Blood MedMNIST

This dataset consists of 17,092 images of blood cells across eight different groups. The images were captured using a Blood Cell Microscope and underwent preprocessing to achieve a resolution of 3×28×28. This dataset was selected for its low-resolution biomedical images, which are well-suited for quantum generative models in the Noisy Intermediate-Scale Quantum (NISQ) era.

To train the models with this dataset, you need to specify the path to the .npz file that contains the data. Additionally, for more information about the required parameters, run `python train.py --helpblood`.

Link to the dataset: [http://braintumorsegmentation.org/](https://medmnist.com/)

_References:_

[1] Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni. Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification." Scientific Data, 2023.

[2] Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

### BraTS dataset

This dataset contains 750 brain volumes with four types of MRI scans: Fluid-attenuated inversion recovery (FLAIR), T1, T1 with gadolinium contrast, and T2. Each volume is composed of 155 slices, each at a resolution of 240×240. This dataset was chosen for its more complex images, which are valuable for studying how models perform with higher-resolution data.

To train the models with this dataset, specify the path to the folder containing the NIfTI files. For more information on the required parameters, run `python train.py --helpbrain`.

Link to the dataset: http://braintumorsegmentation.org/

_References:_

[3] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

[4] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

[5] S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)


## Quantum GANs
### Quantum Patch GAN

The initial tests were conducted using the Quantum Patch GAN architecture proposed in [6]. This hybrid model employs a patch-based method, where the generator comprises multiple sub-generators, each implemented with parameterized quantum circuits (PQCs) and responsible for generating a specific portion of the image. The discriminator, is a classical fully connected neural network. This approach significantly reduces the number of qubits needed, enabling the model to handle higher-resolution images more effectively.

The model code was sourced from [7] and adapted to the datasets. Additional modifications were made to integrate the model with this repository, including functions for saving and loading the model, as well as storing generated images. For more information on the required parameters, run `python train.py --helpPatchGAN`.


_References:_

[6] Huang, H. L., Du, Y., Gong, M., Zhao, Y., Wu, Y., Wang, C., ... & Pan, J. W. (2021). Experimental quantum generative adversarial networks for image generation. Physical Review Applied, 16(2), 024051.

[7] James Ellis. Quantum GANs. Aug. 2024. url: https://pennylane.ai/qml/demos/tutorial_quantum_gans/

### QC-GAN

Another promising hybrid quantum-classical GAN architecture is the QC-GAN [8]. Unlike the patch method, this architecture captures global features across the entire image. The generator combines a quantum circuit with a classical neural network layer, which performs the nonlinear operations and removes the need for ancillary qubits. As with previous models, the discriminator is classical, consisting of a fully connected neural network. For more information on the required parameters, run `python train.py --helpQCGAN`.

_References:_

[8] Shu, R., Xu, X., Yung, M. H., & Cui, W. (2024). Variational Quantum Circuits Enhanced Generative Adversarial Network. arXiv preprint arXiv:2402.01791.


### PQWGAN

Building on the Quantum PatchGAN approach, an improved version was proposed in [9]. This architecture enhances convergence by minimizing the Wasserstein distance during training. Like PatchGAN, it utilizes multiple sub-generators; however, the quantum circuit differs in structure. The discriminator is a classical fully connected network with three hidden layers and outputs a real value that approximates the Wasserstein distance, which is then used to optimize the model. 

The code for this model was sourced from [10] and adapted to integrate seamlessly with this repository. For more information on the required parameters, run `python train.py --helpPQWGAN`.

_References:_

[9] Tsang, S. L., West, M. T., Erfani, S. M., & Usman, M. (2023). Hybrid quantum-classical generative adversarial network for high resolution image generation. IEEE Transactions on Quantum Engineering.

[10] Jasontslxd. GitHub - jasontslxd/PQWGAN: Code to run and test the PQWGAN framework. url: https://github.com/jasontslxd/PQWGAN

## Classical GANs
### GAN

For the classical counterparts of the models, a traditional GAN was trained. This architecture consists of fully connected neural networks with three hidden layers for both the generator and the discriminator. The code was sourced from [11] and modified to support training on different datasets, enable model saving and loading, and ensure compatibility with this repository. Additionally, for more information about the required parameters, run `python train.py --helpGAN`.


_References:_

[11] Real Python. Generative adversarial networks: Build your first models. Aug. 2024. url: https://realpython.com/generative-adversarial-networks/.

### WGAN

To compare the results of PQWGAN with its classical counterpart, the WGAN-GP architecture was trained. Proposed by [12] and serving as the basis for PQWGAN, WGAN-GP aims to enhance training stability by incorporating a gradient penalty for the critic. The code was sourced from [13] and modified to support training on various datasets, enable model saving and loading, and ensure compatibility with the repository. Additionally, for more information about the required parameters, run `python train.py --helpWGAN`.

_References:_

[12] Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. C. (2017). Improved training of wasserstein gans. Advances in neural information processing systems, 30.

[13] Eriklindernoren. PyTorch-GAN. url: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/wgan_gp.

## Quantum Denoising Diffusion Models
### Q-Dense and Q-Dense Directed

Q-Dense is another architecture utilized in this project. It is a quantum denoising model introduced in [14], where dense quantum circuits are trained to perform denoising. These circuits are termed "dense" due to the significant entanglement they create between qubits, making them somewhat analogous to fully connected networks in classical machine learning. Building on this architecture, the authors also proposed a guided version of the model that incorporates image class embeddings to enhance training. In this improved model, an additional ancilla qubit encodes the label information.

The Q-Dense code was adapted from [15], with modifications to support training on various datasets, enable model saving and loading, and ensure compatibility with this repository. This adaptation is based on software originally created by Michael Kölle and is licensed under the MIT License: Copyright (c) 2024 Michael Kölle

For details on required parameters, run `python train.py --help QDense / --help QDensedi`.

_References:_

[14] Michael Kölle et al. “Quantum Denoising Diffusion Models”. In: arXiv preprint arXiv:2401.07049 (2024).

[15] Michaelkoelle. GitHub - michaelkoelle/quantum-diffusion: Quantum Denoising Diffusion Models. url: https://github.com/michaelkoelle/quantum-diffusion.


## Classical Denoising Diffusion Models
### Denoising Diffusion Probabilistic architecture

Finally, to compare the results of the denoising diffusion models, a classical counterpart was used using a U-Net based architecture for the denoising process. The code was sourced from [16] and modified to integrate with this repository, support training on various datasets, and enable model saving and loading. Additionally, for more information about the required parameters, run `python train.py --helpDiff`.

_References:_

[16] Brian Pulfer. “Generating images with DDPMs: A PyTorch Implementation”. In: (Mar. 2023). url: https://medium.com/@brianpulfer/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1.

