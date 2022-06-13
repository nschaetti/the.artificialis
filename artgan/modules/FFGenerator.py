
# Imports
import torch.nn as nn


# Create generator blocks
def get_generator_block(input_dim, output_dim):
    r"""
    Function for returning a block of the generator's neural network
    given input and output dimensions.

    :param input_dim: the dimension of the input vector, a scalar
    :param output_dim: the dimension of the output vector, a scalar
    :return: a generator neural network layer, with a linear transformation followed by a batch
    normalization and then a relu activation.
    """
    return nn.Sequential(
        # Fully connected layer
        nn.Linear(input_dim, output_dim),
        # Batch normalisation
        nn.BatchNorm1d(output_dim),
        # ReLU activation function
        nn.ReLU(inplace=True)
    )
# end get_generator_blocks


# Create generator class
class FFGenerator(nn.Module):
    r"""
    Generator Class.

    :param z_dim: the dimension of the noise vector, a scalar
    :param im_dim: the dimension of the images, fitted for the dataset used, a scalar
    (MNIST images are 28 x 28 = 784 so that is your default).
    :param hidden_dim: the inner dimension, a scalar.
    """

    def __init__(
            self,
            z_dim=10,
            im_dim=784,
            hidden_dim=128
    ):
        # Call to Module
        super(FFGenerator, self).__init__()

        # Build the neural network
        self.gen = nn.Sequential(
            # Linear, bach norm, ReLU
            get_generator_block(z_dim, hidden_dim),
            # Linear, bach norm, ReLU
            get_generator_block(hidden_dim, hidden_dim * 2),
            # Linear, bach norm, ReLU
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            # Linear, bach norm, ReLU
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            # Linear + Sigmoid
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )
    # end init

    # Forward pass
    def forward(self, noise):
        """Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.

        :param noise: a noise tensor with dimensions (n_samples, z_dim)
        :return:
        """
        return self.gen(noise)
    # end forward

    # Needed for grading
    def get_gen(self):
        """
        Returns:
            the sequential model
        """
        return self.gen
    # end get_gen

# end FFGenerator

