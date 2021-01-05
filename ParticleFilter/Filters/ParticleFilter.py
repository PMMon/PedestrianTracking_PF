# ==Imports==
import os, sys
import argparse
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from ParticleFilter.Filters.MotionModel.MotModel import *
from ParticleFilter.Filters.ObservationModel.ColorHistogram import *
from ParticleFilter.Filters.ObservationModel.InvariantMoments import *
from ParticleFilter.Filters.Resampling.SystematicResamp import *

# ====================
#  Particle Filter
# ====================

class PF:
    def __init__(self, args):
        self.args = args

        # Motion model
        self.motion = Motion()

        # Resampling method
        if self.args.resampling == "SYS":
            self.resampling = SYS()
        elif self.args.resampling == "VAN":
            self.resampling = SYS()
        else:
            raise ValueError("Invalid resampling method %s! Please choose either SYS or VAN.", self.args.resampling)


    def initparticles(self, image_size, rng=0):
        """
        Initialize particles
        :return:
        """
        img_height = image_size[0]
        img_width = image_size[1]

        # draw N particles from uniform distribution
        x = np.random.randint(0, 0.9*img_width, self.args.N)
        y = np.random.randint(0, 0.9*img_height, self.args.N)

        dx = np.random.uniform(0.0, 1.0, self.args.N) - 1
        dy = np.random.uniform(0.0, 1.0, self.args.N) - 1

        Hx = np.random.uniform(0.2, 1.0, self.args.N) * (img_width-x)
        Hy = np.random.uniform(0.2, 1.0, self.args.N) * (img_height-y)
        Hx = Hx.astype(int)
        Hy = Hy.astype(int)

        s = np.random.uniform(0.8, 1.2, self.args.N)

        particles = np.array([x, y, dx, dy, Hx, Hy, s])
        weights = 1/self.args.N * np.ones(self.args.N)

        return particles, weights


    def observationmodel(self, type, nr):
        """
        Defines observation model of PF
        :param type: Type of observation model
        :param nr: Helps to differ between CM and other OM
        :return: Observation model
        """
        if type == "MMT" or (type == "CM" and nr == 1):
            return InvMom(self.args)
        elif type == "CLR" or (type == "CM" and nr == 2):
            return ClrHisto(self.args)
        else:
            raise ValueError("Invalid OM (Oberservation Model): %s. Please choose:\n- CLR\n- MMT\n- CM", type)


    def centerparticles(self, state_vectors):
        """
        Centers Particles from upper left corner of BBox to center of BBox
        :param state_vectors: Numpy array with particle states - size: [7 x N]
        :return: Centered state vectors
        """
        new_state = state_vectors.copy()
        new_state[0, :] = (state_vectors[0, :] + 0.5 * state_vectors[4, :]).astype(int)
        new_state[1, :] = (state_vectors[1, :] + 0.5 * state_vectors[5, :]).astype(int)
        return new_state
