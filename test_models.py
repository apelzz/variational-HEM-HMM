import unittest
from models import MultivariateGaussian
import numpy
from scipy.stats import multivariate_normal
from unittest import TestCase

class testModels(TestCase):
    """Tests the various models defined in models.py."""

    def test_init_multivariate_gaussian(self):
        """Tests if the instance variables are correctly set for MultivariateGaussian()."""
        mean = [0,0]
        cov = [[1,0],[0,1]]
        test_norm = MultivariateGaussian(mean=mean, cov=cov)

        assert numpy.allclose(test_norm.mean, mean)
        assert numpy.allclose(test_norm.cov, cov)

    def test_multivariate_gaussian_pdf(self):
        """Tests if MultivariateGaussian() returns the same pdf values as its scipy counterpart."""
        mean = [0,0]
        cov = [[1,0],[0,1]]
        x = [[1.1, 2.2], [-99, 201]]
        test_pdf = MultivariateGaussian(mean=mean, cov=cov).pdf(x)
        scipy_pdf = multivariate_normal.pdf(x=x, mean=mean, cov=cov)

        assert numpy.allclose(test_pdf, scipy_pdf)



if __name__ == "__main__":
    unittest.main()
