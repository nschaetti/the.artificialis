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

# Import
import math
import pygame
import numpy as np
import numpy.linalg as lin
from Unit import Unit


class Fish(Unit):
    r"""A fish.
    """

    # Color
    COLOR = (0, 150, 255)

    # Protection distance
    PROTECT_RADIUS = 20

    # Attraction weight
    ATTRACTION_WEIGHT = 0.05
    REPULSION_WEIGHT = 5

    # SPEED
    SPEED = 1.0

    def __init__(
            self,
            position: np.array,
            angle: float
    ):
        r"""

        Args:
            position:
            angle:
        """
        super(Fish, self).__init__(position, angle)

        self.xd = self.yd = 0
    # end __init__

    # Rule 1 and 2
    def rule12(self, neighbor):
        r"""Rule 1 and 2.

        Args:
            neighbor:

        Returns:

        """
        # Angle with neighbor
        vd = neighbor.position - self.position
        return math.atan2(vd[1], vd[0])
    # end rule12

    # Update the position
    def update(self, bound_x: int, bound_y: int, *args, **kwargs):
        r"""

        Returns:

        """
        # Neighbor
        neig = kwargs['neighbor']

        # Get direction
        rad = self.rule12(neig)

        rad_diff = rad - self.angle

        if rad_diff > math.pi: rad_diff -= 2.0 * math.pi
        if rad_diff < -math.pi: rad_diff += 2.0 * math.pi

        self.xd = math.cos(rad)
        self.yd = math.sin(rad)

        # Distance
        neig_d = lin.norm(neig.position - self.position)

        # Compute weight, max log
        x_dist = neig_d - self.PROTECT_RADIUS
        max_log = math.log(max(bound_x, bound_y))
        weight = (math.log(neig_d / self.PROTECT_RADIUS) / max_log)
        if weight > 0: weight *= self.ATTRACTION_WEIGHT
        if weight < 0: weight *= self.REPULSION_WEIGHT

        # Update angle
        self.angle += rad_diff * weight

        if self.angle > 2.0 * math.pi:
            self.angle -= 2.0 * math.pi
        # end if

        if self.angle < 0:
            self.angle += 2.0 * math.pi
        # end if

        # Update position
        super(Fish, self).update(bound_x, bound_y)
    # end update

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

        # Draw the head
        pygame.draw.circle(
            screen,
            self.COLOR,
            (int(self.x + head_x), int(self.y + head_y)),
            self.RADIUS
        )

        # Draw the queue
        pygame.draw.circle(
            screen,
            self.COLOR,
            (int(self.x - head_x), int(self.y - head_y)),
            self.RADIUS
        )

        # Draw direction to neighbor
        """pygame.draw.line(
            screen,
            (255, 0, 0),
            (self.xi, self.yi),
            (self.xi + self.xd * 20, self.yi + self.yd * 20)
        )"""

        # Draw self direction
        """pygame.draw.line(
            screen,
            (255, 255, 0),
            (self.xi, self.yi),
            (self.xi + math.cos(self.angle) * 20, self.yi + math.sin(self.angle) * 20)
        )"""

        # Protection zone
        """pygame.draw.circle(
            screen,
            (255, 0, 0),
            (int(self.x), int(self.y)),
            self.PROTECT_RADIUS,
            width=1
        )"""
    # end draw

# end Fish
