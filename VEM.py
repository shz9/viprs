import numpy as np


class vem_prs(object):

    def __init__(self, gdl, theta=None):
        """
        :param gdl: An instance of GWAS data loader
        :param theta: 
        """

        if theta is None:
            self.theta = self.initialize_theta()
        else:
            self.theta = theta

    @staticmethod
    def initialize_theta():
        return {
            'sigma_beta': np.random.uniform(),
            'sigma_epsilon': np.random.uniform(),
            'pi': np.random.uniform()
        }

    def e_step(self):
        """
        In the E-step, we update the variational parameters
        for each SNP.
        :return:
        """
        pass

    def m_step(self):
        """
        In the M-step, we update the global parameters of
        the model.
        :return:
        """
        pass

    def objective(self):
        pass

    def iterate(self, max_iter=50):

        for i in range(max_iter):
            pass

