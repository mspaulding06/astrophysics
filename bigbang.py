#!/usr/bin/env python
#
# Sets up a bunch of masses and lets them move under the influence of their
# mutual gravity.
# Masses are put in random positions within a sphere originally. An additional
# force is applied which simulates an equal density distributed over the
# rest of the universe outside this initial sphere.
#
# Stars interacting gravitationally
# Program uses numpy arrays for high speed computations
#
# NOTE: Adapted from script in edX course Greatest Unsolved Mysteries of the Universe
#       This script has been modified to support Python 3.7 and newer versions of the
#       numpy and vpython libraries, as well as other improvements
#
# Original Script Link:
# https://prod-edxapp.edx-cdn.org/assets/courseware/v1/f6b996eb15f7797ce9486fd71b4c2365/asset-v1:ANUx+ANU-ASTRO1x+2T2016+type@asset+block/structureformation.py
#


import math
import sys
from dataclasses import dataclass

from numpy import add, array, less_equal, sqrt, newaxis, sort, nonzero, identity
from numpy.random import random, uniform
from vpython import scene, sphere, vec, rate


@dataclass
class SimulationContext:
    stars: int
    G: int = 6.7e-11
    Msun: int = 1.5e30
    Rsun: int = 3e8
    L: int = 4e10
    h0: int = 1.0e-5  # Hubble Constant
    p0: int = 0.0 * Msun * 100000.0


class Simulation(object):
    def __init__(self, ctx: SimulationContext):
        self.ctx = ctx
        self.__init_scene__()
        self.__init_stars__()

    def __init_scene__(self):
        scene.width = 1320
        scene.height = 830
        scene.range = self.ctx.L
        scene.forward = vec(-1, -1, -1)

    def __init_stars__(self):
        self.stars = []
        self.pos_list = []
        self.p_list = []
        self.m_list = []
        self.r_list = []

        for _ in range(self.ctx.stars):
            v = self.ctx.L * Simulation.random_vector() * (random() ** (1.0 / 3.0))
            x = v[0]
            y = v[1]
            z = v[2]
            r = self.ctx.Rsun
            col0 = vec(uniform(0.7, 1.0), uniform(0.7, 1.0),
                       uniform(0.7, 1.0))
            self.stars.append(sphere(pos=vec(x, y, z), radius=r, color=col0))
            mass = self.ctx.Msun
            px = self.ctx.p0 * uniform(-1, 1)
            py = self.ctx.p0 * uniform(-1, 1)
            pz = self.ctx.p0 * uniform(-1, 1)
            self.pos_list.append((x, y, z))
            self.p_list.append((px, py, pz))
            self.m_list.append(mass)
            self.r_list.append(r)

    @staticmethod
    def random_direction():
        # Generates random direction on sky.
        # RA (in radians)
        # Dec (in radians)
        ra = 2.0 * math.pi * random()
        dec = math.acos(2.0 * random() - 1.0) - 0.5 * math.pi
        return ra, dec

    @staticmethod
    def random_vector():
        # Generates a randomly orientated unit vector.
        theta, phi = Simulation.random_direction()
        z = math.sin(phi)
        x = math.cos(phi) * math.sin(theta)
        y = math.cos(phi) * math.cos(theta)
        return array([x, y, z])

    def run(self):
        pos = array(self.pos_list)
        p = array(self.p_list)
        m = array(self.m_list)
        m.shape = (self.ctx.stars, 1)  # Numeric Python: (1 by Nstars) vs. (Nstars by 1)
        radius = array(self.r_list)
        vcm = sum(p) / sum(m)  # velocity of center of mass
        p = p - m * vcm  # make total initial momentum equal zero

        dt = 50.0
        pos = pos + (p / m) * (dt / 2.0)  # initial half-step
        num_hits = 0
        L = self.ctx.L

        while True:
            rate(50)
            L *= 1.0 + self.ctx.h0 * dt
            # strength of force to allow for external mass
            con = 1.0 * self.ctx.G * self.ctx.stars * self.ctx.Msun / L ** 3
            # Compute all forces on all stars
            r = pos - pos[:, newaxis]  # all pairs of star-to-star vectors
            for i in range(self.ctx.stars):
                r[i, i] = 1e6  # otherwise the self-forces are infinite
            rmag = sqrt(add.reduce(r ** 2, -1))  # star-to-star scalar distances
            hit = less_equal(rmag, radius + radius[:, newaxis]) - identity(self.ctx.stars)
            # 1,2 encoded as 1 * stars + 2
            hit_list = sort(nonzero(hit.flat)[0]).tolist()
            F = self.ctx.G * m * m[:, newaxis] * r / rmag[:, :, newaxis] ** 3  # all force pairs

            for i in range(self.ctx.stars):
                F[i, i] = 0  # no self-forces
            p = p + sum(F, 1) * dt + pos * con * dt * m

            # Having updated all momenta, now update all positions
            pos += (p / m) * dt

            # Expand universe
            pos *= 1.0 + self.ctx.h0 * dt

            # Update positions of display objects; add trail
            for i in range(self.ctx.stars):
                v = pos[i]
                self.stars[i].pos = vec(v[0], v[1], v[2])

            # If any collisions took place, merge those stars
            for hit in hit_list:
                i, j = divmod(hit, self.ctx.stars)  # decode star pair
                if not (self.stars[i].visible or self.stars[j].visible):
                    continue
                # m[i] is a one-element list, e.g. [6e30]
                # m[i,0] is an ordinary number, e.g. 6e30
                new_pos = (pos[i] * m[i, 0] + pos[j] * m[j, 0]) / (m[i, 0] + m[j, 0])
                new_mass = m[i, 0] + m[j, 0]
                new_p = p[i] + p[j]
                new_radius = self.ctx.Rsun * ((new_mass / self.ctx.Msun) ** (1.0 / 3.0))
                i_set, j_set = i, j
                if radius[j] > radius[i]:
                    i_set, j_set = j, i
                self.stars[i_set].radius = new_radius
                m[i_set, 0] = new_mass
                pos[i_set] = new_pos
                p[i_set] = new_p
                self.stars[j_set].visible = 0
                p[j_set] = vec(0, 0, 0)
                m[j_set, 0] = self.ctx.Msun * 1e-30  # give it a tiny mass
                num_hits += 1
                pos[j_set] = (10.0 * L * num_hits, 0, 0)  # put it far away


if __name__ == "__main__":
    stars = 200
    if len(sys.argv) > 1:
        stars = int(sys.argv[1])
    Simulation(SimulationContext(stars)).run()
