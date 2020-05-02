import os
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets.samples_generator import make_blobs


class GMM:
    def __init__(self, X, n_clusters, iterations):
        self.X = X
        self.n_clusters = n_clusters
        self.iterations = iterations
        self.mu = None
        self.pi = None
        self.cov = None
        self.XY = None
        # print("X:",)
        # print(self.X[:2, :])
    """Define a function which runs for iterations, iterations"""
    def run(self):
        self.reg_cov = 1e-6 * np.identity(self.X.shape[1])

        """ 1. Set the initial mu, covariance and pi values"""

        # This is a n x m matrix since we assume n sources (n Gaussians) where each has m dimensions
        self.mu = np.random.randint(np.min(self.X[:, 0]), np.max(self.X[:, 0]),
                                    size=(self.n_clusters, self.X.shape[1]))

        # Since we need a n x m x m covariance matrix for each cluster since we have m features -->
        # We create symmetric covariance matrices with ones on the diagonal
        self.cov = np.zeros((self.n_clusters, self.X.shape[1], self.X.shape[1]))
        for dim in range(len(self.cov)):
            np.fill_diagonal(self.cov[dim], 1)

        self.pi = np.ones(self.n_clusters) / self.n_clusters  # Fractions

        log_likelihoods = []  # List of log likelihoods per iterations to check the convergence

        for i in range(self.iterations):

            """E-Step"""
            z_ik = np.zeros((self.X.shape[0], self.n_clusters))  # Latent Variables
            for m, co, p, z in zip(self.mu, self.cov, self.pi, range(self.X.shape[1])):
                co += self.reg_cov
                mn = multivariate_normal(mean=m, cov=co)
                z_ik[:, z] = p * mn.pdf(self.X) / np.sum([
                    pi_k * multivariate_normal(mean=mu_k, cov=cov_k).pdf(X) for pi_k, mu_k, cov_k in
                    zip(self.pi, self.mu, self.cov+self.reg_cov)], axis=0)

            """ M-Step"""
            # Calculate the new mean vector and new covariance matrices,
            # based on the probable membership of the single x_i to classes c --> z_ik

            self.mu = []
            self.cov = []
            self.pi = []

            for k in range(self.n_clusters):
                m_k = np.sum(z_ik[:, k], axis=0)
                # Compute pi_new
                self.pi.append(m_k/np.sum(z_ik))
                # Compute mu_new (mean new)
                mu_k = (1/m_k) * np.sum(self.X * z_ik[:, k].reshape(self.X.shape[0], 1), axis=0)
                self.mu.append(mu_k)
                # compute sigma_mu (covariance matrix) on the new meam
                self.cov.append(((1/m_k) * np.dot((z_ik[:, k].reshape(self.X.shape[0], 1)*np.subtract(self.X, mu_k)).T,
                                                  np.subtract(self.X, mu_k))) + self.reg_cov)  # covariance m x m
                # I am not sure (1/m_k) * is correct or not

                # self.cov.append(((1/m_k)*np.dot((np.array(z_ik[:, k]).reshape(len(self.X),1)*(self.X-mu_k)).T,
                #                                 (self.X-mu_k)))+self.reg_cov)

            """Log likelihood"""
            log_likelihoods.append(
                np.log(np.sum([k*multivariate_normal(self.mu[i], self.cov[j]).pdf(X) for k, i, j in
                               zip(self.pi, range(len(self.mu)), range(len(self.cov)))])))

    """Predict the membership of an unseen, new datapoint"""
    def predict(self, Y):
        prediction = []
        for m, c in zip(self.mu, self.cov):
            prediction.append(multivariate_normal(mean=m, cov=c).pdf(Y)/ np.sum([
                multivariate_normal(mean=mean, cov=cov).pdf(Y) for mean, cov in zip(self.mu, self.cov)]))

        return prediction


if __name__ == '__main__':

    # 0. Create dataset
    X, Y = make_blobs(cluster_std=1.5, random_state=20, n_samples=500, centers=2)
    # Stratch dataset to get ellipsoid data
    X = np.dot(X, np.random.RandomState(0).randn(2, 2))

    GMM = GMM(X, 2, 50)
    GMM.run()
    print(GMM.predict([[0.5, 0.5]]))
    print("done!")