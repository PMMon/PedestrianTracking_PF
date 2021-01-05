# ==Imports==
import os, sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# ===============================================
#  Super class for various observation models
# ===============================================

class ObservModel:
    def __init__(self, args):
        self.args = args

    def normpdf(self, x, mu, sigma):
        """
        Calculates normal distribution
        :param x: Variable of PDF
        :param mu: Mean
        :param sigma: Variance
        """
        return 1/(sigma*(2*np.pi)**0.5)*np.exp(-1*(x-mu)**2/(2*sigma**2))
