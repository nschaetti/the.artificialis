#!/usr/bin/env python
# coding: utf-8
# This file is part of the the.artificialis distribution (https://github.com/nschaetti/the.artificialis).
# Copyright (c) 2022 Nils Schaetti.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

# Imports
import pygame
import numpy as np
import matplotlib.pyplot as plt


# Parameters
width = 1024
height = 1024
n_frames = 10000 + 1
n_conv = 80
n_fishes = 100

# random walk
random_walk = np.random.randn(n_fishes, 2, n_frames + n_conv)
random_walk = np.cumsum(random_walk, axis=2)
random_grad = random_walk[:, :, 1:] - random_walk[:, :, :-1]
conv_kernel = np.array([1.0 / n_conv] * n_conv)
fish_grad = np.apply_along_axis(lambda m: np.convolve(conv_kernel, m, 'same'), axis=2, arr=random_grad)

# Init pygame
pygame.init()

# Set up drawing window
screen = pygame.display.set_mode([width, height])

running = True

# Position
# pos = np.array([width // 2, height // 2], dtype=np.float)
pos = np.vstack(
    (
        np.random.randint(0, width, n_fishes).astype(np.float32),
        np.random.randint(0, height, n_fishes).astype(np.float32),
    )
).T

tindex = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # end if
    # end for

    screen.fill((0, 0, 0))

    for fish_i in range(n_fishes):
        pygame.draw.circle(screen, (255, 255, 255), (int(pos[fish_i, 0]), int(pos[fish_i, 1])), 4)

        # Update position
        pos[fish_i] += fish_grad[fish_i, :, tindex % (n_frames - 1)]

        if pos[fish_i, 0] >= width: pos[fish_i, 0] = 0
        if pos[fish_i, 1] >= height: pos[fish_i, 1] = 0
        if pos[fish_i, 0] < 0: pos[fish_i, 0] = width - 1
        if pos[fish_i, 1] < 0: pos[fish_i, 1] = height - 1
    # end for

    tindex += 1

    pygame.display.flip()
# end while

pygame.quit()
