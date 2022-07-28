from hmmlearn.hmm import GMMHMM
import numpy as np
from scipy.stats._multivariate import multivariate_normal_frozen

class MultivariateGaussian(multivariate_normal_frozen):
    """Extends a scipy.stats.multivariate_normal object."""
    def __init__(self, mean, cov):
        """
        Initializes a multivariate Gaussian object.

        :param mean: The mean vector (or scalar) of the Gaussian distribution.
        :type mean: array_like (numerical)

        :param cov: The covariance matrix of the Gaussian distribution.
                    The dimension should be len(mean) x len(mean).
        :type cov: array_like (numerical)
        """
        super().__init__(mean=mean, cov=cov)
        self.dimension = len(mean)

class GaussianMixture(object):
    """Implements a Gaussian Mixture Model."""

    def __init__(self, components, weights):
        """
        Initializes a Gaussian Mixture object.

        :param component: A list of individual MultivariateGaussian objects.
        :type components: array_like (MultivariateGaussian)

        :param weights: A list of weights of the corresponding Gaussian component.
                        The weights should sum to one and have the same length as `components`.
        :type weights: array_like (numerical)
        """
        self.components = components
        self.weights = weights
        self.n_components = len(components)
        self.dimension = components[0].dimension

    def _pdf(self, x):
        """
        Helper function to compute the pdf value for a single coordinate.

        :param x: The input coordinate (could be a scalar or a list of scalars).
        :type x: array_like or single numerical

        :return: The pdf value for the single coordinate.
        :rtype: numerical
        """
        return np.sum([self.weights[i] * self.components[i].pdf(x) for i in range(len(self.components))])

    def pdf(self, x):
        """
        Computes the pdf value for a list of input coordinates.

        :param x: A list of input coordinates for whom the pdf values will be calculated.
        :type x: array_like (numerical)

        :return: A list of output pdf values (same length as x).
        :rtype: numpy.array (numerical)
        """
        return np.array([self._pdf(xi) for xi in x])
    
    def get_means(self):
        """Returns the set of all means with shape (n_components, dimension)."""
        flat_means = np.array([g.mean for g in self.components])
        return flat_means.reshape((self.n_components, self.dimension))

    def get_covs(self):
        """Returns the set of all covariances with shape (n_components, dimension, dimension)."""
        flat_covs = np.array([g.cov for g in self.components])
        return flat_covs.reshape((self.n_components, self.dimension, self.dimension))
    

class HiddenMarkovModel(GMMHMM):
    """Extends a hmmlearn.hmm.GMMHMM object (Gaussian Mixture HMM)."""
    def __init__(self, components, startprob, transmat):
        super().__init__()
        # GMMHMM parameters
        self.n_components = len(components)
        self.n_mix = components[0].n_components
        self.covariance_type = "full"
        self.startprob_ = startprob
        self.transmat_ = transmat

        self.weights_ = np.array([gmm.weights for gmm in components]).\
                        reshape((self.n_components,self.n_mix))
        self.means_ = np.array([gmm.get_means() for gmm in components]).\
                      reshape(self.n_components,self.n_mix,components[0].dimension)
        self.covars_ = np.array([gmm.get_covs() for gmm in components]).\
                       reshape(self.n_components,self.n_mix,components[0].dimension,components[0].dimension)

        # Child class specific parameters
        self.components = components


class HiddenMarkovMixtureModel(object):

    def __init__(self):
        pass
