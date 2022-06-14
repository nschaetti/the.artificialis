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

from Unit import Unit
from  Fish import Fish


# Parameters
width = 1920
height = 1080
n_frames = 10000
n_fishes = 100

# Init pygame
pygame.init()

# Set up drawing window
screen = pygame.display.set_mode([width, height])

# Create the fishes
fish_population = list()
for _ in range(n_fishes):
    fish_population.append(Fish.create_random(width, height))
    # fish_population.append(Unit(np.array([512, 512]), 0))
# end for

# Always running
running = True

# Init time index
tindex = 0

# Run simulation
while running:
    # Stop if asked
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # end if
    # end for

    # Fill screen with black
    screen.fill((0, 0, 0))

    # For each fish
    for fish in fish_population:
        # Draw the fish
        fish.draw(screen)

        # Get nearest neighbor
        neig, neig_d = fish.get_nearest_neighbor(fish_population)

        # Update the position of the fish
        fish.update(width, height, neighbor=neig)
    # end for

    tindex += 1

    # Flip display
    pygame.display.flip()
# end while

# Quit game
pygame.quit()
