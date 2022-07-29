from operator import mod
import models 
import numpy as np
from scipy.stats import multivariate_normal
import unittest

class TestModels(unittest.TestCase):
    """Tests the various models defined in models.py."""

    def test_init_multivariate_gaussian(self):
        """Tests if the instance variables are correctly set for MultivariateGaussian()."""
        mean = [0,0]
        cov = [[1,0],[0,1]]
        test_norm = models.MultivariateGaussian(mean=mean, cov=cov)

        assert np.allclose(test_norm.mean, mean)
        assert np.allclose(test_norm.cov, cov)

    def test_multivariate_gaussian_pdf(self):
        """Tests if MultivariateGaussian() returns the same pdf values as its scipy counterpart."""
        mean = [0,0]
        cov = [[1,0],[0,1]]
        x = [[1.1, 2.2], [-99, 201]]
        test_pdf = models.MultivariateGaussian(mean=mean, cov=cov).pdf(x)
        scipy_pdf = multivariate_normal.pdf(x=x, mean=mean, cov=cov)

        assert np.allclose(test_pdf, scipy_pdf)

    def test_init_gmm(self):
        """Tests if the instance variables are correctly set for GaussianMixtureModel()."""
        gaussian_1 = models.MultivariateGaussian(mean=[0, 0], cov=[[1, 0], [0, 1]])
        gaussian_2 = models.MultivariateGaussian(mean=[1, 1], cov=[[2, 0], [0, 2]])
        gaussian_3 = models.MultivariateGaussian(mean=[99, 99], cov=[[2, 1], [1, 2]])
        gaussians = [gaussian_1, gaussian_2, gaussian_3]
        weights = [0.7, 0.2, 0.1]
        test_gmm = models.GaussianMixtureModel(gaussians, weights)

        self.assertListEqual(test_gmm.mixtures, gaussians)
        self.assertListEqual(test_gmm.weights, weights)
        self.assertEqual(test_gmm.n_mix, len(gaussians))
        self.assertEqual(test_gmm.n_features, gaussian_1.n_features)

    def test_gmm_pdf(self):
        """Tests if GaussianMixtureModel.pdf correctly calculates the pdf values for input coordinates."""
        x = [[1.1, 2.2], [-1, -0.5], [99, 100]]
        gaussian_1 = models.MultivariateGaussian(mean=[0, 0], cov=[[1, 0], [0, 1]])
        gaussian_2 = models.MultivariateGaussian(mean=[1, 1], cov=[[2, 0], [0, 2]])
        gaussian_3 = models.MultivariateGaussian(mean=[99, 99], cov=[[2, 1], [1, 2]])
        gaussians = [gaussian_1, gaussian_2, gaussian_3]
        weights = [0.7, 0.2, 0.1]

        test_gmm = models.GaussianMixtureModel(gaussians, weights)
        test_pdf = test_gmm.pdf(x)
        scipy_pdf = [
            0.7 * gaussian_1.pdf(xi) + \
            0.2 * gaussian_2.pdf(xi) + \
            0.1 * gaussian_3.pdf(xi) \
            for xi in x
        ]

        assert np.allclose(test_pdf, scipy_pdf)

    def test_gmm_get_means(self):
        """Tests if GaussianMixtureModel.get_means() correctly returns the set of means in the right shape."""
        gaussian_1 = models.MultivariateGaussian(mean=[0, 0], cov=[[1, 0],[ 0, 1]])
        gaussian_2 = models.MultivariateGaussian(mean=[1, 1], cov=[[2, 0], [0, 2]])
        gaussian_3 = models.MultivariateGaussian(mean=[99, 99], cov=[[2, 1], [1, 2]])
        gaussians = [gaussian_1, gaussian_2, gaussian_3]
        weights = [0.7, 0.2, 0.1]
        test_gmm = models.GaussianMixtureModel(gaussians, weights)

        expected_means = np.array([[0, 0],
                                   [1, 1],
                                   [99, 99]])
        
        self.assertEqual(test_gmm.get_means().shape, expected_means.shape)
        assert np.allclose(test_gmm.get_means(), expected_means)

    def test_gmm_get_covs(self):
        """Tests if GaussianMixtureModel.get_covs() correctly returns the set of covariances in the right shape."""
        gaussian_1 = models.MultivariateGaussian(mean=[0,0,0], cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        gaussian_2 = models.MultivariateGaussian(mean=[1,1,1], cov=[[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        gaussian_3 = models.MultivariateGaussian(mean=[99,99,99], cov=[[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        gaussians = [gaussian_1, gaussian_2, gaussian_3]
        weights = [0.7, 0.2, 0.1]
        test_gmm = models.GaussianMixtureModel(gaussians, weights)

        expected_covs = np.array([[[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]], 
                                   [[2, 0, 0],
                                    [0, 2, 0],
                                    [0, 0, 2]],
                                   [[2, 1, 1],
                                    [1, 2, 1],
                                    [1, 1, 2]]])
        
        self.assertEqual(test_gmm.get_covs().shape, expected_covs.shape)
        assert np.allclose(test_gmm.get_covs(), expected_covs)

    def test_init_hmm(self):
        """Tests if the instance variables are correctly set for HiddenMarkovModel()."""
        gaussian_1 = models.MultivariateGaussian(mean=[0, 0], cov=[[1, 0],[ 0, 1]])
        gaussian_2 = models.MultivariateGaussian(mean=[1, 1], cov=[[2, 0], [0, 2]])
        gaussian_3 = models.MultivariateGaussian(mean=[99, 99], cov=[[2, 1], [1, 2]])
        gaussian_4 = models.MultivariateGaussian(mean=[101, 101], cov=[[1, 0], [0, 1]])

        gmm_1 = models.GaussianMixtureModel([gaussian_1, gaussian_2], [0.5, 0.5])
        gmm_2 = models.GaussianMixtureModel([gaussian_3, gaussian_4], [0.7, 0.3])
        gmm_3 = models.GaussianMixtureModel([gaussian_1, gaussian_3], [0.9, 0.1])
        gmm_4 = models.GaussianMixtureModel([gaussian_2, gaussian_4], [0.2, 0.8])
        
        transmat = [[0.7, 0.1, 0.2, 0.0],
                    [0.1, 0.3, 0.4, 0.2],
                    [0.2, 0.4, 0.1, 0.1],
                    [0.0, 0.2, 0.1, 0.7]]
        test_hmm = models.HiddenMarkovModel(
            components=[gmm_1, gmm_2, gmm_3, gmm_4],
            startprob=[0.25] * 4,
            transmat=transmat
        )

        self.assertEqual(test_hmm.n_components, 4)
        self.assertEqual(test_hmm.n_mix, gmm_1.n_mix)
        self.assertEqual(test_hmm.covariance_type, "full")
        self.assertListEqual(test_hmm.components, [gmm_1, gmm_2, gmm_3, gmm_4])
        self.assertEqual(test_hmm.n_features, 2)
        self.assertListEqual(test_hmm.startprob_, [0.25] * 4)
        assert np.allclose(test_hmm.transmat_, transmat)
        assert np.allclose(
            test_hmm.weights_, 
            [[0.5, 0.5],[0.7, 0.3],[0.9, 0.1],[0.2, 0.8]]
        )
        assert np.allclose(
            test_hmm.means_,
            [[[0, 0],[1, 1]], [[99, 99],[101, 101]],
             [[0, 0], [99, 99]], [[1, 1], [101, 101]]]
        )
        assert np.allclose(
            test_hmm.covars_,
            [[[[1, 0], [0, 1]], [[2, 0], [0, 2]]],
             [[[2, 1], [1, 2]], [[1, 0], [0, 1]]],
             [[[1, 0], [0, 1]], [[2, 1], [1, 2]]],
             [[[2, 0], [0, 2]], [[1, 0], [0, 1]]]]
        )


if __name__ == "__main__":
    unittest.main()
