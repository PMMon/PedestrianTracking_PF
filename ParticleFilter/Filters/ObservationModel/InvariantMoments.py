# ==Imports==
import os, sys
import numpy as np
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from ParticleFilter.Filters.ObservationModel.ObsModel import ObservModel

# =======================================================================
#  Observation Model of Particle Filter - Based on invariant moments
# =======================================================================

class InvMom(ObservModel):
    def __init__(self, args):
        super(InvMom, self).__init__(args)

    def initialization(self, state_vectors, frame):
        """
        Initialize state vector for invariant moments
        :param state_vector: Numpy array with set of state vectors - size: [7 x N]
        :param frame: Current frame
        """
        N = state_vectors.shape[1]

        # Iterate over particles and calculate invariant moments for bounding box
        for n in range(N):
            moment_vector = np.array([])

            particle_state = state_vectors[:, n]
            self.xl = max(int(particle_state[0] + 0.0*particle_state[4]), 0)
            self.yl = max(int(particle_state[1] + 0.0*particle_state[5]), 0)
            self.xr = min(int(particle_state[0] + 1.0*particle_state[4] + 1), frame.shape[1])
            self.yr = min(int(particle_state[1] + 0.7*particle_state[5] + 1), frame.shape[0])

            target = frame[self.yl:self.yr, self.xl:self.xr, :]

            # Convert target to black color
            gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

            # Detect black jacket of pedestrian as trackable shape
            mask_black = cv2.inRange(gray, 0, 30)

            #moment_vector = np.append(moment_vector, self.calcinvariants(mask_black))

            huMoments = cv2.HuMoments(cv2.moments(mask_black))
            moment_vector = np.append(moment_vector, huMoments.reshape(1, 7))

            if n == 0:
                p = moment_vector
            else:
                p = np.vstack((p, moment_vector))

        return p


    def distance(self, phi_t, phi_c):
        """
        Calculates distance between measurement vectors of target and observation
        :param phi_t: Feature vector of target - size: [1 x 512]
        :param phi_c: Feature vector of particles - size: [N x 512]
        """
        N = phi_c.shape[0]
        d = np.array([])

        for n in range(N):
            r1 = phi_c[n, :]
            r2 = phi_t

            # Shift feature vectors to positve domain
            min_value = min(np.min(r1), np.min(r2))
            r1 = r1 + abs(min_value) + 1e-15
            r2 = r2 + abs(min_value) + 1e-15
            r = abs(np.divide((-r1 + r2), (r1 + r2)))
            d = np.append(d, np.array([1 / len(r) * np.sum(r)]), axis=0)

        return d


    def likelihood(self, phi_t, phi_c, mu, sigma):
        """
        Calculates distance between measurement vectors of target and observation and computes respective distribution of weights
        :param phi_t: measurement vector of target - size: [1 x 7]
        :param phi_c: measurement vector of observation - size: [N x 7]
        :return weights: weights of particles - size: [N x 1]
        """
        d = self.distance(phi_t, phi_c)

        mu_vec = mu * np.ones(d.shape)
        sigma_vec = sigma * np.ones(d.shape)

        weights = np.array(list(map(self.normpdf, d, mu_vec, sigma_vec)))
        if np.sum(weights, axis=0) >= 1e-14:
            weights = 1/np.sum(weights, axis=0) * weights
        else:
            weights = 1/self.args.N * np.ones(self.args.N)

        return weights


    def normalize(self, h):
        """
        Normalizes array h
        :param h: Numpy array
        """
        if np.sum(h) == 0:
            return h
        else:
            return h/np.sum(h)


    def moment(self, a, b, x, y, I):
        xa = x**a
        yb = y**b
        mab = np.dot(xa, I).dot(yb)
        return mab


    def centeredmoment(self, a, b, x, y, xmu, ymu, I):
        xa = (x-xmu)**a
        yb = (y-ymu)**b
        mab = np.dot(xa, I).dot(yb)
        return mab


    def calcmoments(self, I):
        #x = np.arange(1, I.shape[1] + 1)
        #y = np.arange(1, I.shape[0] + 1)
        x = np.arange(self.xl, self.xr)
        y = np.arange(self.yl, self.yr)
        m00 = self.moment(0, 0, y, x, I)
        m10 = self.moment(1, 0, y, x, I)
        m01 = self.moment(0, 1, y, x, I)
        if m00 <= 1:
            print("Invalid m00!")
            print("m00: " + str(m00))
            print("xl: " + str(self.xl))
            print("xr: " + str(self.xr))
            print("x: " + str(x))
            print("x: " + str(y))
            print("size I: " + str(I.shape))
        xmu = m10/m00
        ymu = m01/m00
        mu00 = self.centeredmoment(0, 0, y, x, xmu, ymu, I)
        mu02 = self.centeredmoment(0, 2, y, x, xmu, ymu, I)
        mu11 = self.centeredmoment(1, 1, y, x, xmu, ymu, I)
        mu20 = self.centeredmoment(2, 0, y, x, xmu, ymu, I)
        mu30 = self.centeredmoment(3, 0, y, x, xmu, ymu, I)
        mu21 = self.centeredmoment(2, 1, y, x, xmu, ymu, I)
        mu12 = self.centeredmoment(1, 2, y, x, xmu, ymu, I)
        mu03 = self.centeredmoment(0, 3, y, x, xmu, ymu, I)

        return mu00, mu02, mu11, mu20, mu30, mu21, mu12, mu03


    def eta(self, a, b, muab, mu00):
        gamma = (a+b)*0.5+1
        return muab/(mu00**gamma)


    def calcetas(self, I):
        mu00, mu02, mu11, mu20, mu30, mu21, mu12, mu03 = self.calcmoments(I)
        eta02 = self.eta(0, 2, mu02, mu00)
        eta11 = self.eta(1, 1, mu11, mu00)
        eta20 = self.eta(2, 0, mu20, mu00)
        eta30 = self.eta(3, 0, mu30, mu00)
        eta21 = self.eta(2, 1, mu21, mu00)
        eta12 = self.eta(1, 2, mu12, mu00)
        eta03 = self.eta(0, 3, mu03, mu00)

        return eta02, eta11, eta20, eta30, eta21, eta12, eta03


    def calcinvariants(self, I):
        eta02, eta11, eta20, eta30, eta21, eta12, eta03 = self.calcetas(I)
        im1 = eta20 + eta02
        im2 = (eta20 - eta02)**2 + 4*eta11**2
        im3 = (eta30 - 3*eta12)**2 + (3*eta21-eta03)**2
        im4 = (eta30 + eta12)**2 + (eta03+eta21)**2
        im5 = (eta30 - 3*eta12)*(eta30+eta12)*((eta30 + eta12)**2 - 3*(eta21 + eta03)**2) + (3*eta21 - eta03)*(eta21 + eta03) * (3*(eta30 + eta12)**2 - (eta21 + eta03)**2)
        im6 = (eta20 - eta02)*((eta30 + eta12)**2 - (eta21 + eta03)**2) + 4*eta11 * (eta30 + eta12)*(eta21+eta03)
        im7 = -(3*eta21 - eta03)*(eta30+eta12)*((eta30+eta12)**2 - 3*(eta21+eta03)**2)+(eta30-3*eta12)*(eta21+eta03)*(3*(eta30+eta12)**2-(eta21+eta03)**2)

        return np.array([im1, im2, im3, im4, im5, im6, im7])
