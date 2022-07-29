"""
This script instantiates objects relevant to HMM clustering.

Relationship among the objects below:
(1) Each MultivariateGaussian is an individual Gaussian distribution.
(2) Each GaussianMixture (GMM) is a weighted collection of the single Gaussians.
    Each GMM describes the distribution of a particular "state".
(3) Each HiddenMarkovModel (HMM) is a collection of the GMMs.
    An HMM describes how the observations represents transitions among different states.
    The most likely "state" for a particular observation depends on both the previous state and 
    whichever state results in higher likelihood of the current observation.
(4) A HiddenMarkovMixtureModel (H3M) is a collection of HMMs.
    To accurately describe the state transition behavior of a particular subject,
    multiple trials are needed,
    and each HMM describes the state transition behavior of a particular trial.

Meaning of common object attributes:
(1) n_feature:  number of features tracked for each observation.
                E.g., n_feature=2 if both x- and y-coordinates are tracked.
(2) n_mix:  number of mixtures (multivariate Gaussian objects) in each Gaussian Mixture Model. 
            For simplicity, assume n_mix is constant for all GMMs (in all states).
(3) n_component:    number of components in each Hidden Markov Model. Essentially, 
                    number of states tracked by each HMM and
                    number of GMM contained in each HMM
                    because each GMM describes the distribution of each state.
"""
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
        self.n_features = len(mean)


class GaussianMixtureModel(object):
    """Implements a Gaussian Mixture Model."""

    def __init__(self, mixtures, weights):
        """
        Initializes a Gaussian Mixture object.

        :param mixtures: A list of individual MultivariateGaussian objects.
        :type mixtures: list (MultivariateGaussian)

        :param weights: A list of weights of the corresponding Gaussian component.
                        The weights should sum to one and have the same length as `components`.
        :type weights: array_like (numerical)
        """
        self.mixtures = mixtures
        self.weights = weights
        self.n_mix = len(mixtures)
        self.n_features = mixtures[0].n_features

    def _pdf(self, x):
        """
        Helper function to compute the pdf value for a single coordinate.

        :param x: The input coordinate (could be a scalar or a list of scalars).
        :type x: array_like or single numerical

        :returns: The pdf value for the single coordinate.
        :rtype: numerical
        """
        return np.sum([self.weights[i] * self.mixtures[i].pdf(x) for i in range(self.n_mix)])

    def pdf(self, x):
        """
        Computes the pdf value for a list of input coordinates.

        :param x: A list of input coordinates for whom the pdf values will be calculated.
        :type x: array_like (numerical)

        :returns: A list of output pdf values (same length as x).
        :rtype: numpy.array (numerical)
        """
        return np.array([self._pdf(xi) for xi in x])
    
    def get_means(self):
        """Returns the set of all means with shape (n_mix, n_feature)."""
        flat_means = np.array([g.mean for g in self.mixtures])
        return flat_means.reshape((self.n_mix, self.n_features))

    def get_covs(self):
        """Returns the set of all covariances with shape (n_mix, n_feature, n_feature)."""
        flat_covs = np.array([g.cov for g in self.mixtures])
        return flat_covs.reshape((self.n_mix, self.n_features, self.n_features))
    

class HiddenMarkovModel(GMMHMM):

    """Extends a hmmlearn.hmm.GMMHMM object (Gaussian Mixture HMM)."""

    def __init__(self, components, startprob, transmat):
        """
        Initializes a Hidden Markov Model object.
        For this algorithm, assume all GMM components have the same number of mixtures (n_mix).

        :param components: The list of GMM objects describing the state distributions.
        :type components: list (of GaussianMixtureModel())

        :param startprob: The vector of probabilities of starting in each of the states.
                          len(startprob) = len(components)
        :type startprob: array_like (numerical)

        :param transmat: The transition matrix describing the probabilities of transitioning between states.
                         transmat.shape = (len(components) x len(components))
        :type transmat: array_like (numerical)
        """
        # GMMHMM initialization parameters
        kwargs = {
            "n_components": len(components),
            "n_mix": components[0].n_mix, # Assume all GMMs have the same number of mixtures for simplicity
            "covariance_type": "full",
        }
        super().__init__(**kwargs)

        # Child class specific parameters
        self.components = components
        self.n_features = components[0].n_features

        # Set (GMMHMM) instance variables
        self.startprob_ = startprob 
        self.transmat_ = transmat
        self.weights_ = np.array([gmm.weights for gmm in components]).\
                        reshape((self.n_components, self.n_mix))
        self.means_ = np.array([gmm.get_means() for gmm in components]).\
                      reshape(self.n_components, self.n_mix, self.n_features)
        self.covars_ = np.array([gmm.get_covs() for gmm in components]).\
                       reshape(self.n_components, self.n_mix, self.n_features, self.n_features)


class HiddenMarkovMixtureModel(object):

    def __init__(self):
        pass
