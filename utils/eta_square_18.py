"""
This script defines a function that computes eta hat as in (18) in Coviello et al., (2014).
Intuitively, each eta_i_beta_j_rho is the responsibility matrix between two Gaussian components in
    (1) the ith HMM in the base corresponding to state beta,
    (2) the jth HMM in the reduced corresponding to state rho.
Further, each eta_i_beta_j_rho_l_m is the probability that an observation 
    from GMM component m of the ith HMM in the base
    corresponds to GMM component l of the jth HMM in the reduced. 
eta_hat(i_beta_j_rho_l_m) is the value that maximizes eta_i_beta_j_rho_l_m.
"""

def _ELogLikelihood_Gaussian_wrt_Gaussian(Gaussian_b, Gaussian_r):
    """
    Computes the expected log-likelihood of a Gaussian (reduced) w.r.t. another Gaussian (base).
    Originally developed in Penny and Roberts (2000).

    :param Gaussian_b: The Gaussian component from a GMM in the base model.
    :type Gaussian_b: scipy.stats.multivariate_normal

    :param Gaussian_r: The Gaussian component from a GMM in the reduced model.
    :type Gaussian_r: scipy.stats.multivariate_normal

    :return: The expected log-likelihood of `Gaussian_r` w.r.t. `Gaussian_b`.
    :rtype: float
    """
    return 10.0

def get_eta_hat_18(GaussianMixture_b, m_b, GaussianMixture_r, l_r):
    """
    Computes eta hat as defined in Coviello et al. (2014) (18).
    
    :param GaussianMixture_b: The Gaussian Mixture Model of interest in an HMM in the base H3M.
    :type GaussianMixture_b: models.GaussianMixture

    :param m_b: The index of the Gaussian model of interest among the list in `GaussianMixture_b`.
                0 <= m_b < number of Gaussian models in `GaussianMixture_b`.
    :type m_b: int

    :param GaussianMixture_r: The Gaussian Mixture Model of interest in an HMM in the reduced H3M.
    :type GaussianMixture_r: models.GaussianMixture

    :param l_r: The index of the Gaussian model of interest among the list in `GaussianMixture_b`.
                0 <= m_b < number of Gaussian models in `GaussianMixture_b`.
    :type l_r: int
    
    :return: Eta hat for the specified Gaussian's in base and in reduced, respectively.
    :rtype: float.
    """
    return 42.0