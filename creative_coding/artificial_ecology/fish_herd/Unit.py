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
import sys
import pygame
import random
import numpy as np
import numpy.linalg as lin
import math


# A unit
class Unit:

    # Unit color
    COLOR = (255, 255, 255)

    # Speed of the unit
    SPEED = 0.3

    # Radius of the unit
    RADIUS = 4.0

    # Constructor
    def __init__(
            self,
            position: np.array,
            angle: float
    ):
        r"""

        Args:
            position: Initial position of the unit
            angle: Angle of the unit
        """
        # Properties
        self.position = position
        self.angle = angle
    # end __init__

    # region PROPERTIES

    # X position
    @property
    def x(self):
        r"""X position.

        Returns:

        """
        return self.position[0]
    # end x

    # Set X
    @x.setter
    def x(self, value):
        r"""Set X.

        Args:
            value:

        Returns:

        """
        self.position[0] = value
    # end x

    # X position
    @property
    def xi(self):
        r"""X position.

        Returns:

        """
        return int(self.position[0])
    # end x

    # Y position
    @property
    def y(self):
        r"""Y position.

        Returns:

        """
        return self.position[1]
    # end y

    # Set Y
    @y.setter
    def y(self, value):
        r"""Set Y.

        Args:
            value:

        Returns:

        """
        self.position[1] = value
    # end y

    # Y position
    @property
    def yi(self):
        r"""Y position.

        Returns:

        """
        return int(self.position[1])
    # end y

    # endregion PROPERTIES

    # region PUBLIC

    # Draw the unit
    def draw(self, screen):
        r"""Draw the unit.

        Returns:

        """
        # Draw the body
        pygame.draw.circle(
            screen,
            self.COLOR,
            (int(self.x), int(self.y)),
            self.RADIUS
        )

        # Position of the head
        head_x = math.cos(self.angle) * self.RADIUS
        head_y = math.sin(self.angle) * self.RADIUS
    # end draw

    # Update the position
    def update(self, bound_x: int, bound_y: int, *args, **kwargs):
        r"""

        Returns:

        """
        # Differences
        dx = math.cos(self.angle) * self.SPEED
        dy = math.sin(self.angle) * self.SPEED

        # Update position
        self.x += dx
        self.y += dy

        # Check boundaries
        if self.x >= bound_x: self.x = 0
        if self.y >= bound_y: self.y = 0
        if self.x < 0: self.x = bound_x - 1
        if self.y < 0: self.y = bound_y - 1
    # end update

    # Get distance to the unit
    def distance(self, v):
        r"""Get the distance o the unit.

        Args:
            v:

        Returns:

        """
        return lin.norm(self.position - v)
    # end distance

    # Get nearest neighbor
    def get_nearest_neighbor(self, neighbors):
        r"""Get nearest neighbor.

        Returns:

        """
        min_dist = sys.maxsize
        nearest = None
        for neig in neighbors:
            if neig is not self:
                neig_d = self.distance(neig.position)
                if neig_d < min_dist:
                    nearest = neig
                    min_dist = neig_d
                # end if
            # end if
        # end for

        return nearest, neig_d
    # end get_nearest_neighbor

    # endregion PUBLIC

    # region STATIC

    # Create a unit with random position
    @classmethod
    def create_random(cls, x_max: int, y_max: int) -> 'Unit':
        r"""Create a unit with random position.

        Args:
            x_max:
            y_max:

        Returns:

        """
        return cls(
            position=np.array([
                random.randint(0, x_max),
                random.randint(0, y_max)
            ]).astype(np.float32),
            angle=random.random() * 2.0 * math.pi
        )
    # end create_random

    # endregion STATIC

    # region OVERRIDE

    def __str__(self):
        r"""To string

        Returns:

        """
        return f"Unit(x={self.x}, y={self.y}, angle={self.angle})"
    # end __str__

    def __repr__(self):
        r"""To string

        Returns:

        """
        return f"Unit(x={self.x}, y={self.y}, angle={self.angle})"
    # end __str__

    # endregion OVERRIDE

# end Unit

