import argparse
import math
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from einops import rearrange, repeat

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import Compose, ToTensor, Lambda

from datasets import BloodDataset, BrainDataset
from models import PatchGAN, QCGAN, PQWGAN, GAN, WGAN, Diffusion, QDenseUndirected, QDenseDirected, MyDDPM, MyUNet



def compute_gradient_penalty(critic, real_samples, fake_samples, device, classic=False):
    """Calculates the gradient penalty loss for WGAN GP"""
    
    batch_size, C, W, H = real_samples.shape
    epsilon = torch.rand(batch_size, 1, 1, 1).repeat(1, C, W, H).to(device)
    interpolated_images = (epsilon * real_samples + ((1 - epsilon) * fake_samples))
    interpolated_scores = critic(interpolated_images)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        inputs=interpolated_images,
        outputs=interpolated_scores,
        grad_outputs=torch.ones_like(interpolated_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.shape[0], -1)
    if classic:
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    else:
        gradient_penalty = torch.mean((1. - torch.sqrt(1e-8+torch.sum(gradients**2, dim=1)))**2)
    return gradient_penalty

def add_normal_noise_multiple(data, tau, decay_mod=1.0):
    """
    Distorting the data by sampling from one normal distribution with a fixed mean.
    Adding the noise with different weights to the tensor.
    """
    if data.dim() == 1:
        data = data.unsqueeze(0)
    batch, pixels = data.shape
    noise = torch.normal(mean=0.5, std=0.2, size=(batch, pixels)).to(
        data.device
    )  # normal distribution
    data_expanded = repeat(data, "batch pixels -> tau batch pixels", tau=tau)
    noise_expanded = repeat(noise, "batch pixels -> tau batch pixels", tau=tau)
    noise_weighting = torch.linspace(0, 1, tau).to(data.device) ** decay_mod
    noise_weighting = noise_weighting / noise_weighting.max()  # normalize
    noise_weighting = repeat(noise_weighting, "tau -> tau batch 1", batch=batch)
    noisy_data = (
        data_expanded * (1 - noise_weighting) + noise_expanded * noise_weighting
    )
    noisy_data = noisy_data.clamp(0, 1)
    noisy_data = rearrange(noisy_data, "tau batch pixels -> (batch tau) pixels")
    return noisy_data


# Function for training different GAN models (classical and quantum)
def trainingGANs(model, discriminator, generator, dataloader, device, batch_size, lr_G, lr_D, image_size,out_dir, n_data_qubits=0, save_interval=5, optimizer='SGD', n_epochs=50, lambda_gp=10):
    # Generator and discriminator initialization
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    # Initialize training variables
    counter=0
    start_epoch=0
    saved_initial=False
    checkpoint_path=out_dir+'checkpoint.pth'
    # Optimisers initialization
    if optimizer=='SGD':
        optD = optim.SGD(discriminator.parameters(), lr=lr_D)
        optG = optim.SGD(generator.parameters(), lr=lr_G)
    elif optimizer=='Adam':
        optD = optim.Adam(discriminator.parameters(), lr=lr_D)
        optG = optim.Adam(generator.parameters(), lr=lr_G)
    # Check if a saved checkpoint exists; if so, load it to resume training from that point
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        generator.load_state_dict(checkpoint['generator_state_dict'])
        optD.load_state_dict(checkpoint['optD_state_dict'])
        optG.load_state_dict(checkpoint['optG_state_dict'])
        start_epoch=checkpoint['epoch']
        counter=checkpoint['counter']
        saved_initial=True
        print("Checkpoint loaded.")
    else:
        print("No checkpoint found. Starting from scratch.")
    # Define the loss function as Binary Cross-Entropy
    criterion = nn.BCELoss()
    # Real and fake labels for the discriminator
    real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
    fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)
    # List of quantum models
    quantum=['QC-GAN', 'PatchGAN', 'PQWGAN']
    # Generate fixed noise for tracking the generator's output during training
    if model in quantum:
        fixed_noise = torch.rand(batch_size, n_data_qubits, device=device)
        if model != 'PQWGAN':
            fixed_noise = fixed_noise * math.pi / 2
    else:
        fixed_noise=torch.randn((batch_size, 100)).to(device=device)
    # Training loop
    for epoch in range(start_epoch, n_epochs):
        for i, (data, _) in enumerate(dataloader):
            # Continue training from the point where the last checkpoint was saved
            if i > counter or counter==0:
                if not saved_initial:
                    # Save the first set of real images 
                    save_image(data, os.path.join(out_dir, 'real_samples.png'), nrow=5)
                    saved_initial=True

                if model not in ['PQWGAN', 'WGAN']:
                    data = data.reshape(batch_size, image_size[0] * image_size[1])

                real_data = data.to(device)
                
                # Generate random noise for the generator
                if model in quantum:
                    noise = torch.rand(batch_size, n_data_qubits, device=device) * math.pi / 2
                else:
                    noise=torch.randn((batch_size, 100)).to(device=device)
                # Generate fake data using the generator
                fake_data = generator(noise).to(device)

                # Training the discriminator
                if model=='PQWGAN' or model=='WGAN':
                    optD.zero_grad()
                    outD_real = discriminator(real_data).view(-1)
                    outD_fake = discriminator(fake_data.detach()).view(-1)
                    # Compute gradient penalty for WGAN-GP
                    if model=='WGAN':
                        gradient_penalty = compute_gradient_penalty(discriminator, real_data, fake_data, device, classic=True)
                    else:
                        gradient_penalty = compute_gradient_penalty(discriminator, real_data, fake_data, device)
                    # Compute the loss for the discriminator using Wasserstein distance
                    errD=-torch.mean(outD_real) + torch.mean(outD_fake) + lambda_gp * gradient_penalty
                    wasserstein_distance = torch.mean(outD_real) - torch.mean(outD_fake)
                    errD.backward()
                    optD.step()
                else:
                    discriminator.zero_grad()
                    outD_real = discriminator(real_data).view(-1)
                    if model=='QC-GAN':
                      fake_data=fake_data.view(1, image_size[0]* image_size[1])
                    outD_fake = discriminator(fake_data.detach()).view(-1)
                    # Calculate the real and fake losses using binary cross-entropy
                    errD_real = criterion(outD_real, real_labels)
                    errD_fake = criterion(outD_fake, fake_labels)
                    # Backpropagate the error
                    errD_real.backward()
                    errD_fake.backward()
                    errD = errD_real + errD_fake
                    optD.step()

                # Training the generator
                if model=='PQWGAN':
                    noise = torch.rand(batch_size, n_data_qubits, device=device)
                    fake_data = generator(noise).to(device)
                    optG.zero_grad()
                    outD_fake = discriminator(fake_data).to(device)
                    errG=-torch.mean(outD_fake).to(device)
                    errG.backward()
                    optG.step()
                elif model=='WGAN':
                    noise=torch.randn((batch_size, 100)).to(device=device)
                    fake_data = generator(noise).to(device)
                    outD_fake = discriminator(fake_data).to(device)
                    errG=-torch.mean(outD_fake).to(device)
                    errG.backward()
                    optG.step()
                else:
                    generator.zero_grad()
                    outD_fake = discriminator(fake_data).view(-1)
                    errG = criterion(outD_fake, real_labels)
                    errG.backward()
                    optG.step()
                # Save the model and generated images at the specified intervals
                if i % save_interval == 0:
                    torch.save({
                        'epoch': epoch,
                        'counter': i,
                        'discriminator_state_dict': discriminator.state_dict(),
                        'generator_state_dict': generator.state_dict(),
                        'optD_state_dict': optD.state_dict(),
                        'optG_state_dict': optG.state_dict(),
                    }, checkpoint_path)
                    print(f'Models and optimizers saved at epoch {epoch} and iteration {i}.')
                    # Save the test images generated at the current checkpoint
                    test_images = generator(fixed_noise).view(batch_size,1,image_size[0],image_size[1]).cpu().detach()
                    if model=='PQWGAN':
                        test_images= ((test_images + 1) / 2).clamp(0, 1)
                    save_image(test_images, os.path.join(out_dir, 'Epoch_{}_Iteration_{}.png'.format(epoch, i)), nrow=5)

                # Print loss values for monitoring the training process
                if i % 3 == 0:
                    if model=='PQWGAN':
                        print(f'Epoch: {epoch},Iteration: {i}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}, Wasserstein Distance: {wasserstein_distance.item()}')
                    else:
                        print(f'Epoch: {epoch},Iteration: {i}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}')
        counter=0

# Function for training different Diffusion models (classical and quantum)
def trainDiff(model, diff, out_dir, lr, n_epochs, dataloader, tau, image_shape, device, save_interval):
    # Define the path for saving checkpoints
    checkpoint_path=out_dir+'checkpoint.pth'
    # Initialize variables
    start_epoch = 0
    counter=0
    saved_initial=False
    # Initialize the optimizer
    opt = torch.optim.Adam(diff.parameters(), lr=lr)
    # Check if a checkpoint file already exists to resume training
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        diff.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        counter=checkpoint['counter']
        saved_initial=True

    if model=='Diffclasi':
        criterion = nn.MSELoss()
    # Initialize the first random input tensor
    first_x = torch.rand(15, 1, image_shape[0], image_shape[1]) * 0.5 + 0.75
    
    # Loop over epochs for training
    for epoch in range(start_epoch, n_epochs):
        epoch_loss = 0.0
        # Loop through the data loader for batches
        for i, (x, y) in enumerate(dataloader):
            if not saved_initial:
                # Save the first set of real images and generated images for reference
                save_image(x, os.path.join(out_dir, 'real_samples.png'), nrow=5)
                saved_initial=True
            # Perform training if the current iteration is valid
            if i > counter or counter==0:
                if model=='Diffclasi':
                    n_steps = diff.n_steps
                    eta = torch.randn_like(x).to(device)
                    t = torch.randint(0, n_steps, (x.shape[0],)).to(device)
                    noisy_imgs = diff(x, t, eta)
                    eta_theta = diff.backward(noisy_imgs, t.reshape(x.shape[0], -1))
                    batch_loss = criterion(eta_theta, eta)
                    opt.zero_grad()
                    batch_loss.backward()
                    opt.step()
                    epoch_loss += batch_loss.item() * len(x) / len(dataloader.dataset)
                else:
                    opt.zero_grad()
                    x=x.view(x.shape[0],-1)
                    y=y.squeeze(1)
                    batch_loss, _ = diff(x=x, y=y, T=tau, verbose=True)
                    epoch_loss += batch_loss.mean()
                    opt.step()
                # Print loss information every three iterations
                if i % 3 == 0:
                    if model=='Diffclasi':
                        print(f'Epoch: {epoch},Iteration: {i}, Loss: {epoch_loss:0.3f}')
                    else:
                        print(f'Epoch: {epoch},Iteration: {i}, Loss: {epoch_loss.item():0.3f}')
                # Save the model and generated images at the specified intervals
                if i % save_interval == 0:
                    torch.save({
                        'epoch': epoch,
                        'counter': i,
                        'model_state_dict': diff.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                    }, checkpoint_path)
                    print(f'Models and optimizers saved at epoch {epoch} and iteration {i}.')
                    # Save the test images generated at the current checkpoint
                    first_x = torch.rand(15, 1, image_shape[0], image_shape[1]) * 0.5 + 0.75
                    diff.eval()
                    outp = diff.sample(first_x=first_x, n_iters=tau * 2, show_progress=False).view(15*(1+(2*tau)),1,image_shape[0],image_shape[1]).cpu().detach()
                    # Normalize output images to the range [0, 1]
                    outp = outp - outp.min() 
                    outp = outp / outp.max() 
                    save_image(outp, os.path.join(out_dir, 'Epoch_{}_Iteration_{}.png'.format(epoch, i)), nrow=15)
        counter=0

# Function that provides help information for the Blood MedMNIST dataset
def parse_blood_help():
    print("Help for Blood MedMNIST dataset:")
    print("-d, --dataset, type=str: Specifies the dataset to train on (BloodMNIST).")
    print("-path, --path, type=str: Path to the dataset in a .npz format.")
    print("-ni, --num_imgs, type=int: Number of images used to train (default 3000).")
    print("-shape,--img_shape, type=int: Refers to image shape.")
    print("-cl, --classes, type=int: Specifies the number of classes for training (default 8).")
    print("-bs, --batch_size, type=int: Batch size (default 50)")
    print("Dataset references:")
    print("[1] Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni. Yang, Jiancheng, et al. MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data, 2023.")
    print("[2] Jiancheng Yang, Rui Shi, Bingbing Ni. MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis. IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.)")

# Function that provides help information for the BraTS dataset
def parse_brain_help():
    print("Help for BraTS dataset:")
    print("-d, --dataset, type=str: Specifies the dataset to train on (BRATS).")
    print("-path, --path, type=str: Path to the folder that contains the MRI volumes in a .nii format.")
    print("-ni, --num_imgs, type=int: Number of images used to train (default 3000).")
    print("-shape,--img_shape, type=int: Refers to image shape.")
    print("-cl, --classes, type=int: Specifies the number of classes for training, i.e. the number of MRI modalities to use.")
    print("-sl, --slice, type=int: Slice used in the dataset (default 155//2).")
    print("-bs, --batch_size, type=int: Batch size (default 50)")
    print("Dataset references:")
    print("[1] Ujjwal Baid et al. “The rsna-asnr-miccai brats 2021 benchmark on brain tumor segmentation and radiogenomic classification”. In: arXiv:2107.02314 (2021).")
    print("[2] Bjoern H Menze et al. “The multimodal brain tumor image segmentation benchmark (BRATS)”. In: IEEE transactions on medical imaging 34.10 (2014), pp. 1993–2024.")
    print("[3] Spyridon Bakas et al. “Advancing the cancer genome atlas glioma MRI collections with expert segmentation labels and radiomic features”. In: Scientific data 4.1 (2017), pp. 1–13.")

# Function that provides help information for PatchGAN model
def parse_PatchGAN():
    print("Help for PatchGAN model:")
    print("-m, --model, type=str: Model to train (PatchGAN)")
    print("-dev, --device, type=str: Device used to train (choices= cuda, cpu)")
    print("-q, --n_data_qubits, type=int: Number of data qubits")
    print("-aq, --n_ancilla_qubits, type=int: Number of ancilla qubits (default 1)")
    print("-la, --layers, type=int: Number of layers of the circuit")
    print("-qd, --q_delta, type=int: Q delta (default 1)")
    print("-ep, --n_epochs, type=int: Number of epochs")
    print("-lrG, --lr_G, type=float: Learning rate of the generator")
    print("-lrD, --lr_D, type=float: Learning rate of the discriminator")
    print("-si, --save_interval, type=int: Model save interval")
    print("-opt, --optimizer, type=str: Optimizer")
    print("-o, --out_dir, type=str: Output directory")
    print("Model references:")
    print("[1] He-Liang Huang et al. Experimental Quantum Generative Adversarial Networks for Image Generation. arXiv:2010.06201 (2020).")
    print("[2] Ellis, J. (2024, 7 octubre). Quantum GANs. PennyLane Demos. https://pennylane.ai/qml/demos/tutorial_quantum_gans/")
# Function that provides help information for QCGAN model
def parse_QCGAN():
    print("Help for QCGAN model:")
    print("-m, --model, type=str: Model to train (QC-GAN)")
    print("-dev, --device, type=str: Device used to train (choices= cuda, cpu)")
    print("-q, --n_data_qubits, type=int: Number of data qubits")
    print("-la, --layers, type=int: Number of layers of the circuit")
    print("-qd, --q_delta, type=int: Q delta (default 1)")
    print("-ep, --n_epochs, type=int: Number of epochs")
    print("-lrG, --lr_G, type=float: Learning rate of the generator")
    print("-lrD, --lr_D, type=float: Learning rate of the discriminator")
    print("-si, --save_interval, type=int: Model save interval")
    print("-opt, --optimizer, type=str: Optimizer")
    print("-o, --out_dir, type=str: Output directory")
    print("Model references:")
    print("[1] Runqiu Shu et al. “Variational Quantum Circuits Enhanced Generative Adversarial Network”. In: arXiv preprint arXiv:2402.01791 (2024).")
# Function that provides help information for PQWGAN model
def parse_PQWGAN():
    print("Help for PQWGAN model:")
    print("-m, --model, type=str: Model to train (PQWGAN)")
    print("-dev, --device, type=str: Device used to train (choices= cuda, cpu)")
    print("-q, --n_data_qubits, type=int: Number of data qubits")
    print("-aq, --n_ancilla_qubits, type=int: Number of ancilla qubits (default 1)")
    print("-la, --layers, type=int: Number of layers of the circuit")
    print("-pshape, --patch_shape, type=int: Shape of the patches")
    print("-ch, --channels, type=int: Number of channels of the images (default 1)")
    print("-ep, --n_epochs, type=int: Number of epochs")
    print("-lrG, --lr_G, type=float: Learning rate of the generator")
    print("-lrD, --lr_D, type=float: Learning rate of the discriminator")
    print("-si, --save_interval, type=int: Model save interval")
    print("-opt, --optimizer, type=str: Optimizer")
    print("-o, --out_dir, type=str: Output directory")
    print("-lgp, --lambda_gp, type=int: Lambda gradient penalty (default 10)")
    print("Model references:")
    print("[1] Shu Lok Tsang et al. “Hybrid quantum-classical generative adversarial network for high resolution image generation”. In: IEEE Transactions on Quantum Engineering (2023).")
    print("[2] Jasontslxd. GitHub - jasontslxd/PQWGAN: Code to run and test the PQWGAN framework. url: https://github.com/jasontslxd/PQWGAN.")
# Function that provides help information for a classic GAN model
def parse_GAN():
    print("Help for GAN model:")
    print("-m, --model, type=str: Model to train (GAN)")
    print("-dev, --device, type=str: Device used to train (choices= cuda, cpu)")
    print("-ep, --n_epochs, type=int: Number of epochs")
    print("-lrG, --lr_G, type=float: Learning rate of the generator")
    print("-lrD, --lr_D, type=float: Learning rate of the discriminator")
    print("-si, --save_interval, type=int: Model save interval")
    print("-opt, --optimizer, type=str: Optimizer")
    print("-o, --out_dir, type=str: Output directory")
    print("Model references:")
    print("[1] Real Python. Generative adversarial networks: Build your first models. Aug. 2024. url: https://realpython.com/generative-adversarial-networks/.")
# Function that provides help information for WGAN model
def parse_WGAN():
    print("Help for WGAN model:")
    print("-m, --model, type=str: Model to train (WGAN)")
    print("-dev, --device, type=str: Device used to train (choices= cuda, cpu)")
    print("-ep, --n_epochs, type=int: Number of epochs")
    print("-lrG, --lr_G, type=float: Learning rate of the generator")
    print("-lrD, --lr_D, type=float: Learning rate of the discriminator")
    print("-si, --save_interval, type=int: Model save interval")
    print("-opt, --optimizer, type=str: Optimizer")
    print("-o, --out_dir, type=str: Output directory")
    print("-lgp, --lambda_gp, type=int: Lambda gradient penalty (default 10)")
    print("Model references:")
    print("[1] Eriklindernoren. PyTorch-GAN. url: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/wgan_gp")
# Function that provides help information for Q-Dense model
def parse_QDense():
    print("Help for Q-Dense model:")
    print("-m, --model, type=str: Model to train (Q-Dense)")
    print("-dev, --device, type=str: Device used to train (choices= cuda, cpu)")
    print("-la, --layers, type=int: Number of layers of the circuit")
    print("-ep, --n_epochs, type=int: Number of epochs")
    print("-lr, --lr, type=float: Learning rate")
    print("-tau, --tau, type=int: Tau")
    print("-o, --out_dir, type=str: Output directory")
    print("-si, --save_interval, type=int: Model save interval")
    print("Model references:")
    print("[1] Michael Kölle et al. “Quantum Denoising Diffusion Models”. In: arXiv preprint arXiv:2401.07049 (2024).")
    print("[2] Michaelkoelle. GitHub - michaelkoelle/quantum-diffusion: Quantum Denoising Diffusion Models. url: https : / / github . com / michaelkoelle / quantum-diffusion.")
# Function that provides help information for Q-Dense Directed model
def parse_QDensedi():
    print("Help for Q-Dense Directed model:")
    print("-m, --model, type=str: Model to train (Q-Dense Directed)")
    print("-dev, --device, type=str: Device used to train (choices= cuda, cpu)")
    print("-la, --layers, type=int: Number of layers of the circuit")
    print("-ep, --n_epochs, type=int: Number of epochs")
    print("-lr, --lr, type=float: Learning rate")
    print("-tau, --tau, type=int: Tau")
    print("-o, --out_dir, type=str: Output directory")
    print("-si, --save_interval, type=int: Model save interval")
    print("Model references:")
    print("[1] Michael Kölle et al. “Quantum Denoising Diffusion Models”. In: arXiv preprint arXiv:2401.07049 (2024).")
    print("[2] Michaelkoelle. GitHub - michaelkoelle/quantum-diffusion: Quantum Denoising Diffusion Models. url: https : / / github . com / michaelkoelle / quantum-diffusion.")

# Function that provides help information for classical diffusion models
def parse_diff():
    print("Help for Classical denoising diffusion model model:")
    print("-m, --model, type=str: Model to train (Diffclasi)")
    print("-dev, --device, type=str: Device used to train (choices= cuda, cpu)")
    print("-ep, --n_epochs, type=int: Number of epochs")
    print("-lr, --lr, type=float: Learning rate")
    print("-tau, --tau, type=int: Tau")
    print("-o, --out_dir, type=str: Output directory")
    print("-si, --save_interval, type=int: Model save interval")
    print("-mib, --minb, type=float: Minimum noise level introduced (default 10 ** -4)")
    print("-mab, --maxb, type=float: Maximum noise level introduced (default 0.02)")
    print("Model references:")
    print("[1] Brian Pulfer. “Generating images with DDPMs: A PyTorch Implementation”. In: (Mar. 2023). url: https://medium.com/@brianpulfer/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1.")


if __name__ == "__main__":
    # Initialize the argument parser with a usage description
    parser = argparse.ArgumentParser(usage='%(prog)s', 
                                     description= 'To know the arguments of each dataset type --helpblood or --helpbrain.\n In case you want to know the arguments of each model type --help+model where the models are: PatchGAN, QCGAN, PQWGAN, GAN, WGAN, QDense, QDensedi and Diff.'
    )

    # Add help flags for various datasets and models
    parser.add_argument('--helpblood', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--helpbrain', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--helpPatchGAN', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--helpQCGAN', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--helpPQWGAN', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--helpGAN', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--helpWGAN', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--helpQDense', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--helpQDensedi', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--helpDiff', action='store_true', help=argparse.SUPPRESS)

    # Add arguments specific to datasets
    parser.add_argument("-cl", "--classes", help="classes to train on", type=int, default=8)
    parser.add_argument("-d", "--dataset", help="dataset to train on", type=str, choices=['BloodMNIST', 'BRATS'])
    parser.add_argument("-ni", "--num_imgs", help="Number of images used to train", type=int, default=3000)
    parser.add_argument("-path", "--path", help="Path to the dataset", type=str)
    parser.add_argument("-sl", "--slice", help="Slice used in the BRATS dataset", type=int, default=155//2)
    parser.add_argument("-bs", "--batch_size", help="Batch size", type=int, default=50)
    parser.add_argument("-shape",'--img_shape', type=int, nargs=2, help="Image shape")
    
    # Add shared arguments
    parser.add_argument("-dev", "--device", help="Device used to train", type=str, choices=['cuda', 'cpu'])
    parser.add_argument("-o", "--out_dir", help="output directory", type=str)
    parser.add_argument("-ep", "--n_epochs", help="Number of epochs", type=int)
    parser.add_argument("-m", "--model", help="model to train", type=str, choices=['PatchGAN', 'QC-GAN', 'PQWGAN', 'GAN', 'WGAN', 'Q-Dense', 'Q-Dense Directed', 'Diffclasi'])

    # Arguments specifically for GAN models
    parser.add_argument("-lrG", "--lr_G", help="Learning rate of the generator", type=float)
    parser.add_argument("-lrD", "--lr_D", help="Learning rate of the discriminator", type=float)
    parser.add_argument("-lgp", "--lambda_gp", help="Lambda gradient penalty", type=int, default=10)
    parser.add_argument("-q", "--n_data_qubits", help="number of data qubits", type=int, default=None)
    parser.add_argument("-si", "--save_interval", help="Save interval", type=int)
    parser.add_argument("-opt", "--optimizer", help="Optimizer used", type=str)
    parser.add_argument("-la", "--layers", help="Number of layers", type=int)
    parser.add_argument("-aq", "--n_ancilla_qubits", help="number of ancilla qubits", type=int, default=1)
    parser.add_argument("-qd", "--q_delta", help="Q delta", type=int, default=1)
    parser.add_argument("-ch", "--channels", help="Number of channels of the images", type=int, default=1)
    parser.add_argument("-pshape",'--patch_shape', type=int, nargs=2, help="Patch shape")

    # Arguments for diffusion models
    parser.add_argument("-lr", "--lr", help="Learning rate", type=float)
    parser.add_argument("-tau", "--tau", help="Tau", type=int)
    parser.add_argument("-mib", "--minb", help="Minimum noise level introduced.", type=float, default= 10 ** -4)
    parser.add_argument("-mab", "--maxb", help="Maximum noise level introduced.", type=float, default=0.02)

    # Parse the arguments passed from the command line
    args = parser.parse_args()

    # Call the appropriate help functions based on the provided flags
    if args.helpblood:
        parse_blood_help()

    if args.helpbrain:
        parse_brain_help()

    if args.helpPatchGAN:
        parse_PatchGAN()

    if args.helpQCGAN:
        parse_QCGAN()
    
    if args.helpPQWGAN:
        parse_PQWGAN()

    if args.helpGAN:
        parse_GAN()
    
    if args.helpWGAN:
        parse_WGAN()

    if args.helpQDense:
        parse_QDense()
    
    if args.helpQDensedi:
        parse_QDensedi()

    if args.helpDiff:
        parse_diff()
    
    if args.dataset=='BloodMNIST':
        dataset=BloodDataset(path=args.path, num_imgs=args.num_imgs, image_shape=args.img_shape, transform=None)
        dataloader=DataLoader(dataset, args.batch_size, shuffle=True)

    if args.dataset=='BRATS':
        dataset=BrainDataset(path=args.path, slice=args.slice, num_imgs=args.num_imgs, image_shape=args.img_shape, num_classes=args.classes, transform=None)
        dataloader=DataLoader(dataset, args.batch_size, shuffle=True)
    
    # List of GAN models
    gans=['PatchGAN', 'QC-GAN', 'PQWGAN', 'GAN', 'WGAN']
    # List of diffusion models
    diffusion=['Q-Dense', 'Q-Dense Directed', 'Diffclasi']

    if args.model in gans:
        if args.model=='PatchGAN':
            # Calculate the number of generators based on image shape and qubit configuration
            n_generators=int((args.img_shape[0]*args.img_shape[1])/(2**(args.n_data_qubits-args.n_ancilla_qubits)))
            # Instantiate the PatchGAN model with the appropriate parameters
            model_=PatchGAN(image_shape=args.img_shape, n_generators=n_generators, n_qubits=args.n_data_qubits, n_ancillas=args.n_ancilla_qubits, n_layers=args.layers, q_delta=args.q_delta, device=args.device)
            # Define the generator and discriminator
            generator=model_.generator.to(args.device)
            discriminator=model_.discriminator.to(args.device)
        elif args.model=='QC-GAN':
            model_=QCGAN(args.img_shape, args.n_data_qubits, args.layers, args.q_delta)
            # Define the generator and discriminator
            generator=model_.generator.to(args.device)
            discriminator=model_.discriminator.to(args.device)
        elif args.model=='PQWGAN':
            n_generators=int((args.img_shape[0]*args.img_shape[1])/(2**(args.n_data_qubits-args.n_ancilla_qubits)))
            model_=PQWGAN(args.img_shape, args.channels, n_generators, args.n_data_qubits, args.n_ancilla_qubits, args.layers, args.patch_shape, args.device)
            # Define the generator and discriminator
            generator=model_.generator.to(args.device)
            discriminator=model_.discriminator.to(args.device)
        elif args.model=='GAN':
            model_=GAN(args.img_shape, args.device)
            # Define the generator and discriminator
            generator=model_.generator.to(args.device)
            discriminator=model_.discriminator.to(args.device)
        elif args.model=='WGAN':
            model_=WGAN(args.img_shape, args.device)
            generator=model_.generator.to(args.device)
            discriminator=model_.discriminator.to(args.device)
        # Train the selected GAN model
        trainingGANs(args.model, discriminator, generator, dataloader, args.device, args.batch_size, args.lr_G, args.lr_D, args.img_shape,args.out_dir, args.n_data_qubits, args.save_interval, args.optimizer, args.n_epochs, args.lambda_gp)
    if args.model in diffusion:
        if args.model=='Q-Dense':
            net=QDenseUndirected(args.layers, args.img_shape)
            diff = Diffusion(
                net=net,
                shape=args.img_shape,
                noise_f=add_normal_noise_multiple,
                prediction_goal='data',
                directed=False,
                loss=torch.nn.MSELoss(),
            ).to(args.device)
        elif args.model=='Q-Dense Directed':
            net=QDenseDirected(args.layers, args.img_shape)
            diff = Diffusion(
                net=net,
                shape=args.img_shape,
                noise_f=add_normal_noise_multiple,
                prediction_goal='data',
                directed=True,
                loss=torch.nn.MSELoss(),
            ).to(args.device)
        elif args.model=='Diffclasi':
            # Defining model
            n_steps=args.tau*2
            diff = MyDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=args.minb, max_beta=args.maxb, device=args.device)
        # Train the selected diffusion model
        trainDiff(args.model, diff, args.out_dir, args.lr, args.n_epochs, dataloader, args.tau, args.img_shape, args.device, args.save_interval)
    
