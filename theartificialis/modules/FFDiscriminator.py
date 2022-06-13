
# Imports
import torch.nn as nn


# Create discriminator blocks
def get_discriminator_block(input_dim, output_dim):
    r'''
    Discriminator Block
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation
          followed by an nn.LeakyReLU activation with negative slope of 0.2
          (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html)
    '''
    return nn.Sequential(
        # Linear
        nn.Linear(input_dim, output_dim),
        # Leaky ReLU
        nn.LeakyReLU(inplace=True, negative_slope=0.2)
    )
# end get_discriminator_block


# Disciminator class
class FFDiscriminator(nn.Module):
    r'''
    Discriminator Class

    :param im_dim: the dimension of the images, fitted for the dataset used, a scalar
    (MNIST images are 28x28 = 784 so that is your default)
    :param hidden_dim: the inner dimension, a scalar
    '''

    # Constructor
    def __init__(
            self,
            im_dim: int = 784,
            hidden_dim: int =128
    ):
        # Call to Module
        super(FFDiscriminator, self).__init__()

        # Sequence
        self.disc = nn.Sequential(
            # Linear, batch norm, ReLU
            get_discriminator_block(im_dim, hidden_dim * 4),
            # Linear, batch norm, ReLU
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            # Linear, batch norm, ReLU
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            # Fully connected with output dim 1
            nn.Linear(hidden_dim, 1)
        )
    # end __init__

    # Forward pass
    def forward(self, image):
        r'''
        Function for completing a forward pass of the discriminator: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        return self.disc(image)
    # end forward

    # Needed for grading
    def get_disc(self):
        r"""The sequential model
        """
        return self.disc
    # end get_disc

# end Discriminator
