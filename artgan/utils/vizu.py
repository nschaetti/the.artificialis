
# Imports
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


# Visualise images
def show_tensor_images(title, image_tensor, num_images=25, size=(1, 28, 28)):
    r"""Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.

    :param image_tensor:
    :type image_tensor:
    :param num_images:
    :type num_images:
    :param size:
    :type size:
    """
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.title(title)
    plt.show()
# end show_tensor_images

