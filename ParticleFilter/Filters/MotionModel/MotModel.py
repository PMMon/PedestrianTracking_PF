import os, sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# =====================================
#  Motion Model of Particle Filter
# =====================================

class Motion:
    def __init__(self):
        pass

    def propagate(self, particles, dt, var, img_size, bounds):
        """
        Process step
        :param Particles: N-array of particle states - size: 7 x N
        :param var: Variance for each feature: [var_x, var_y, var_vx, var_vy, var_width, var_height, var_scale] - size: 1 x 7
        :param img_size: Frame size
        :param bounds: Bounds for resulting state vectors: [vx_max, vy_max, minWidth, maxWidth, minHeight, maxHeight, maxScale] - size: 1 x 7
        :return:
        """
        N = particles.shape[1]
        A = np.eye(particles.shape[0])
        A[0, 2] = dt
        A[1, 3] = dt

        # Define sigma and mu for PDF
        sigma = np.eye(particles.shape[0])
        sigma[0, 0] = var[0]
        sigma[1, 1] = var[1]
        sigma[2, 2] = var[2]
        sigma[3, 3] = var[3]
        sigma[4, 4] = var[4]
        sigma[5, 5] = var[5]
        sigma[6, 6] = var[6]

        mu = np.zeros(particles.shape[0])

        # Process
        for i in range(N):
            particle = particles[:, i]
            A[4,4] = particle[6]
            A[5,5] = particle[6]

            particle = np.dot(A, particle) + np.random.multivariate_normal(mu, sigma)

            # ensure that particles still in image
            particle[0] = int(min(max(particle[0], 0), 0.90*img_size[1]))
            particle[1] = int(min(max(particle[1], 0), 0.90*img_size[0]))

            # check bounds
            particle[2] = min(max(particle[2], -bounds[0]), bounds[0])
            particle[3] = min(max(particle[3], -bounds[1]), bounds[1])

            particle[4] = min(max(particle[4], bounds[2]), bounds[3])
            particle[5] = min(max(particle[5], bounds[4]), bounds[5])

            particle[6] = min(max(particle[6], 1-bounds[6]), 1+bounds[6])

            particles[:, i] = particle

        return  particles