
import torch
import torch.nn as nn
from .TemporalGenerator import TemporalGenerator


class VideoGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        # instantiate the temporal generator
        self.temp = TemporalGenerator()

        # create a transformation for the temporal vectors
        self.fast = nn.Sequential(
            nn.Linear(100, 256 * 4**2, bias=False),
            nn.BatchNorm1d(256 * 4**2),
            nn.ReLU()
        )

        # create a transformation for the content vector
        self.slow = nn.Sequential(
            nn.Linear(100, 256 * 4**2, bias=False),
            nn.BatchNorm1d(256 * 4**2),
            nn.ReLU()
        )


        # define the image generator
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        # initialize weights according to the paper
        self.fast.apply(self.init_weights)
        self.slow.apply(self.init_weights)
        self.model.apply(self.init_weights)
    # end __init__

    def init_weights(self, m):
        if type(m) == nn.ConvTranspose2d or type(m) == nn.Linear:
            nn.init.uniform_(m.weight, a=-0.01, b=0.01)
        # end if
    # end init_weights

    def forward(self, x):
        # pass our latent vector through the temporal generator and reshape
        z_fast = self.temp(x).contiguous()
        z_fast = z_fast.view(-1, 100)

        # transform the content and temporal vectors
        z_fast = self.fast(z_fast).view(-1, 256, 4, 4)
        z_slow = self.slow(x).view(-1, 256, 4, 4).unsqueeze(1)
        # after z_slow is transformed and expanded we can duplicate it
        z_slow = torch.cat([z_slow]*16, dim=1).view(-1, 256, 4, 4)

        # concatenate the temporal and content vectors
        z = torch.cat([z_slow, z_fast], dim=1)

        # transform into image frames
        out = self.model(z)

        return out.view(-1, 16, 3, 64, 64).transpose(1, 2)
    # end forward

# end VideoGenerator

