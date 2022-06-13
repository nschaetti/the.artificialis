#!/usr/bin/env python
# coding: utf-8


# Imports
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ArtGAN imports
from artgan.modules import FFGenerator, FFDiscriminator
from artgan.utils import show_tensor_images

torch.manual_seed(0)


# Get noise
def get_noise(
        n_samples: int,
        z_dim: int,
        device: str = 'cpu'
) -> torch.Tensor:
    r"""
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    """
    # Generate n_samples random vectors of dimenson z_dim on device
    return torch.randn(n_samples, z_dim, device=device)


# end get_noise


# Compute disciminator loss
def compute_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    r"""Return the loss of the discriminator given inputs.

    :param gen: the generator model, which returns an image given z-dimensional noise
    :param disc: the discriminator model, which returns a single-dimensional prediction of real/fake
    :param criterion: the loss function, which should be used to compare the discriminator's predictions to the
    ground truth reality of the images (e.g. fake = 0, real = 1).
    :param real: a batch of real images.
    :param num_images: the number of images the generator should produce, which is also the length of the real images.
    :param z_dim: the dimension of the noise vector, a scalar
    :param device: the device type
    :return: disc_loss, a torch scalar loss value for the current batch
    """
    z_vec = get_noise(num_images, z_dim, device=device)
    fake_images = gen(z_vec)
    fake_images = fake_images.detach()
    fake_output = disc(fake_images)
    fake_loss = criterion(fake_output, torch.zeros(num_images, 1, device=device))
    real_output = disc(real)
    real_loss = criterion(real_output, torch.ones(num_images, 1, device=device))
    disc_loss = (fake_loss + real_loss) / 2.0

    return disc_loss


# end get_disc_loss


# Compute generator loss
def compute_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    r"""Return the loss of the generator given inputs.

    :param gen: the generator model, which returns an image given z-dimensional noise
    :param disc: the discriminator model, which returns a single-dimensional prediction of real/fake
    :param criterion: the loss function, which should be used to compare the discriminator's predictions to the
    ground truth reality of the images (e.g. fake = 0, real = 1).
    :param num_images: the number of images the generator should produce, which is also the length of the real images
    :param z_dim: the dimension of the noise vector, a scalar
    :param device: the device type
    :return: gen_loss: a torch scalar loss value for the current batch.
    """
    z_vec = get_noise(num_images, z_dim, device=device)
    fake_images = gen(z_vec)
    disc_output = disc(fake_images)
    gen_loss = criterion(disc_output, torch.ones(num_images, 1, device=device))
    return gen_loss


# end get_gen_loss


# Experiment parameter
criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.00001

# Load MNIST dataset as tensors
dataloader = DataLoader(
    MNIST(
        '.',
        download=True,
        transform=transforms.ToTensor()
    ),
    batch_size=batch_size,
    shuffle=True,
)

# Device
device = 'cuda:0'

# Create the generator and its optimizer
gen = FFGenerator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)

# Create the discriminator and its optimizer
disc = FFDiscriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

# Current step, losses
cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True
gen_loss = False
error = False

# For each epoch
for epoch in range(n_epochs):
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        # Size of the current batch
        cur_batch_size = len(real)

        # Flatten the batch of real images from the dataset
        real = real.view(cur_batch_size, -1).to(device)

        #
        # Update discriminator
        #

        # Zero out the gradients before backprop
        disc_opt.zero_grad()

        # Calculate discriminator loss
        disc_loss = compute_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)

        # Update gradients
        disc_loss.backward(retain_graph=True)

        # Update parameters
        disc_opt.step()

        # For testing purposes, to keep track of the generator weights
        if test_generator:
            old_generator_weights = gen.gen[0][0].weight.detach().clone()

        #
        # Update generator
        #

        # Zero out the gradients before backprop
        gen_opt.zero_grad()

        # Compute generator loss
        gen_loss = compute_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)

        # Update the gradient
        gen_loss.backward(retain_graph=True)

        # Update parameters
        gen_opt.step()

        # For testing purposes,
        # to check that your code changes the generator weights
        if test_generator:
            try:
                assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
            except:
                error = True
                raise Exception("Runtime tests failed")
            # end try
        # end if

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        # Visalise images at each 'display_step'
        if cur_step % display_step == 0 and cur_step > 0:
            # Show step and losses
            print(
                f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}"
            )

            # Create nose vectors
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)

            # Generate images
            fake = gen(fake_noise)

            # Show fakes and reals
            show_tensor_images(f"Fakes at {cur_step}", fake)
            show_tensor_images("Real", real)

            # Reset mean losses
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        # end if

        # Inc step
        cur_step += 1
    # end for
# end for
