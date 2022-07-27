from hmmlearn.hmm import GMMHMM
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
    """Implements a Gaussian Mixture Model"""

    def __init__(self, components, weights):
        """
        Initializes a Gaussian Mixture object

        :param component: A list of individual MultivariateGaussian objects.
        :type components: array_like (MultivariateGaussian)

        :param weights: A list of weights of the corresponding Gaussian component.
                        The weights should sum to one and have the same length as `components`.
        :type weights: array_like (numerical)
        """
        self.components = components
        self.weights = weights
        self.n_components = len(components)

    def pdf(self, x):
        """
        Computes the pdf value for a list of input coordinates.

        :param x: A list of input coordinates for whom the pdf values will be calculated.
        :type x: array_like (numerical)

        :return: A list of output pdf values (same length as x).
        :rtype: array_like (numerical)
        """
        pass
    
    def get_means(self):
        pass

    def get_covs(self):
        pass
    

class HiddenMarkovModel(GMMHMM):
    
    def __init__(self):
        super().__init__()

class HiddenMarkovMixtureModel(object):

    def __init__(self):
        pass
