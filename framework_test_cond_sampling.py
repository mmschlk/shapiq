import numpy as np
from scipy.stats import multivariate_normal

if __name__ == "__main__":

    # Define mean vector and covariance matrix
    mu = np.array([0, 0, 0, 0])  # Mean for X1, X2, X3, X4
    rho = 0  # Correlation coefficient
    # Covariance matrix with correlations
    cov = np.array([[1, rho, rho, rho], [rho, 1, rho, rho], [rho, rho, 1, rho], [rho, rho, rho, 1]])

    # Condition on X3 = 1 and X4 = 1
    x_3_star = 1.0  # Given value of X3
    x_4_star = 1.0  # Given value of X4

    # Partition the mean vector and covariance matrix
    mu_A = mu[:2]  # Mean of X1, X2
    mu_B = mu[2:]  # Mean of X3, X4
    Sigma_AA = cov[:2, :2]  # Covariance of X1, X2
    Sigma_AB = cov[:2, 2:]  # Covariance between X1, X2 and X3, X4
    Sigma_BB = cov[2:, 2:]  # Covariance of X3, X4

    # Compute conditional mean and covariance
    Sigma_BB_inv = np.linalg.inv(Sigma_BB)  # Inverse of the covariance of X3, X4
    mu_conditional = mu_A + Sigma_AB @ Sigma_BB_inv @ np.array([x_3_star, x_4_star] - mu_B)
    Sigma_conditional = Sigma_AA - Sigma_AB @ Sigma_BB_inv @ Sigma_AB.T

    # Sample from the conditional distribution of X1, X2 given X3 = 1 and X4 = 1
    n_samples = 10_000
    conditional_samples = multivariate_normal.rvs(
        mean=mu_conditional, cov=Sigma_conditional, size=n_samples
    )

    # Print some example conditional samples
    sampled = conditional_samples[:10]
    print(sampled)
