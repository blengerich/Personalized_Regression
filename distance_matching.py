# Personalized Logistic Regression
#from numba import jit
import numpy as np
np.set_printoptions(precision=4)
import time

from utils import *
from scipy.spatial import KDTree # To store Z matrix and find closest neighbors.
#from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import concurrent
import concurrent.futures
from numba import jit

import warnings
warnings.filterwarnings("error")


default_log_file='logs/distance_matching_{}.log'.format(
    time.strftime("%Y_%m_%d-%H_%M_%S"))


class DistanceMatching():

    def __init__(self, init_beta,
                f, f_prime,
                gamma, latent_dim, n_neighbors,
                update_ztree_every, calc_dist_errors_every, calc_closest_every,
                rho_beta, rho_beta_prime,
                init_phi_u, psi_u, psi_u_prime,
                intercept, log_dir="./", n_threads=1):
        # Define functions on initialization.
        self.init_beta = init_beta
        self.f = f
        self.f_prime = f_prime
        self.gamma = gamma
        self.latent_dim = latent_dim
        self.n_neighbors = n_neighbors
        self.update_ztree_every = update_ztree_every
        self.calc_dist_errors_every = calc_dist_errors_every
        self.calc_closest_every = calc_closest_every
        self.rho_beta = rho_beta
        self.rho_beta_prime = rho_beta_prime

        self.init_phi_u = init_phi_u
        self.psi_u = psi_u
        self.psi_u_prime = psi_u_prime

        self.intercept = intercept
        self.log_dir = log_dir
        self.n_threads = n_threads
        if self.n_threads > 0:
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.n_threads)
            self.map = lambda x,y: list(self.executor.map(x,y))
        else:
            self.pool = None
            self.map = lambda x,y: list(map(x, y))


    def _check_shapes(self, X, Y, U, dU=None):
        """ Does some basic checks on the shapes on the parameters. """
        N = X.shape[0]
        assert(U.shape[0] == N)
        #P = X.shape[1]
        K = U.shape[1]
        if dU:
            assert(len(dU) == K)
        return N, K


    def make_covariate_distances(self, U, dU, K, N, should_normalize=True):
        # Make fixed distance matrix for co-variates
        t = time.time()
        print("Making Co-Variate Distance Matrix of Size {}x{}x{}".format(N, N, K))
        D = np.zeros((N, N, K))
        #with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_threads) as executor:
        get_dist = lambda i, j: np.array([
            dU[k](U[i, k], U[j, k]) for k in range(K)], dtype="float32")
        for i in range(1, N):
            print("{}\t/{}".format(i, N), end='\r')
            D[i, 0:i, :] = self.map(lambda j: get_dist(i, j), range(i))
        for i in range(1, N):
            for j in range(i):
                D[j, i, :] = D[i, j, :] # could cut memory in half by only storing lists.
        print("Finished making unnormalized version.")
        if should_normalize:
            normalized = np.array([normalize(D[:, :, k]) for k in range(K)])
            # Now the first axis references k. Move it to the back.
            normalized = np.swapaxes(normalized, 0, 1)
            D = np.swapaxes(normalized, 1, 2)
        print("Finished normalizing.")
        print("Took {:.3f} seconds.".format(time.time() - t))
        return D


    def make_covariate_distance_function(self, U, dU, K):
        # Allows lazy calculations of covariate distances, but cannot normalize.
        func = lambda i,j: np.array([dU[k](U[i,k], U[j,k]) for k in range(K)])
        return func

    import matplotlib.pyplot as plt
    def _update_beta_hat(self, iteration=0, plot=False):
        if self.multitask:
            self.beta_hat = np.tensordot(self.Z, self.Q, axes=1)
        else:
            self.beta_hat = self.Z.dot(self.Q) + self.beta_mean
            if plot:
                fig = plt.figure()
                plt.scatter(self.beta_hat[:, 0], self.beta_hat[:, 1], marker='+', alpha=0.5)
                plt.savefig("results/optimization/{}.png".format(iteration))
                plt.show()

    def _calc_losses(self, iteration):
        self._update_beta_hat(iteration, True)
        loss1 = np.sum([
            self.f(self.X[i], self.Y[i], self.beta_hat[i]) for i in range(self.N)])
        loss2 = 0.5*self.gamma*np.mean(
            np.mean(np.square(self.dist_errors), axis=1))
        loss3 = np.mean([self.rho_beta(self.beta_hat[i], i) for i in range(self.N)])
        #loss4 = self.psi_beta(phi_beta)
        loss4 = 0.
        loss5 = self.psi_u(self.phi_u)
        loss6 = 0.#self.psi_nu(self.nu)
        return [loss1, loss2, loss3, loss4, loss5, loss6]


    def _init_data_vars(self, data):
        self.X = data["X"]
        self.Y = data["Y"]
        self.U = data["U"]
        self.delta_U = data["delta_U"]
        self.N, self.K = self._check_shapes(self.X, self.Y, self.U)
        self.P = self.X.shape[1]
        if len(self.init_beta.shape) > 2:
            self.multitask = True
            self.T = self.init_beta.shape[2]
        else:
            self.multitask = False

        self.beta_hat = self.init_beta.copy()
        # Initialize B, Z by PCA of beta_hat
        pca = PCA(n_components=self.latent_dim, whiten=False)
        if self.multitask:
            self.Z = np.zeros((self.N, self.latent_dim))
            self.Q = np.zeros((self.latent_dim, self.P, self.T))
            for t in range(self.T):
                self.Z += pca.fit_transform(self.beta_hat[:, :, t])
                self.Q[:, :, t] = pca.components_
            self.Z /= self.T
        else:
            self.beta_mean = np.mean(self.beta_hat, axis=0)
            print(self.beta_mean.shape)
            self.Z = pca.fit_transform(self.beta_hat)
            self.Q = pca.components_
            print(self.beta_hat)
            print(self.Z.dot(self.Q))
            print(pca.explained_variance_ratio_)
            """
            self.Z = np.random.multivariate_normal(np.zeros(self.latent_dim),
                np.eye(self.latent_dim), self.N)
            self.Q = np.random.uniform(0, 1, size=(self.latent_dim, self.P))
            """
        """
        try:
            z_norms = np.linalg.norm(self.Z, axis=0, ord=1) # normalize features
            self.Z /= np.clip(np.tile(z_norms, (self.N, 1)), 0.01, 100)
            if self.multitask:
                for t in range(self.T):
                    self.Q[:, :, t] *= np.tile(z_norms, (self.P, 1)).T
            else:
                self.Q *= np.tile(z_norms, (self.P, 1)).T
        except RuntimeWarning:
            self.Z = np.random.normal(0, 0.01, size=self.Z.shape)
            self.Q = np.random.normal(0, 0.01, size=self.Q.shape)
        """

        #print(self.Z, self.Q)
        #self.Z = np.random.normal(0, 0.01, size=(self.N, self.latent_dim))
        #self.B = np.random.normal(0, 0.01, size=(self.latent_dim, self.P))#np.linalg.lstsq(self.Z, self.beta_hat)[0]
        #self.nu = 1.#self.init_nu # scalar, no need to copy
        self.phi_u = self.init_phi_u.copy()
        self.beta_prev = self.beta_hat.copy()
        self.phi_u_prev = self.phi_u.copy()

    def _make_z_tree(self):
        self.z_tree = KDTree(self.Z, leafsize=self.kd_leafsize)

    def _init_opt_vars(self, opt_params):
        self.patience  = opt_params["init_patience"]
        self.lr        = opt_params["init_lr"]
        self.lr_decay  = opt_params["lr_decay"]
        self.max_iters = opt_params["max_iters"]
        self.tol       = opt_params["tol"]
        self.prev_loss = np.inf
        self.distances_over_time = []
        self.losses_over_time    = []

        if opt_params.get("neighbors", None):
            print("Neighborhoods supplied. Will use those.")
            self._find_closest_neighbors = lambda: opt_params["neighborhoods"]
        elif opt_params.get("calc_neighbors", False):
            print("No neighborhoods supplied. Will calculate nearest neighbors.")
            self.kd_leafsize = opt_params["kd_leafsize"]
            self._make_z_tree()
            self._find_closest_neighbors = lambda: self.z_tree.query(
                self.Z, k=self.n_neighbors, eps=1.1)[1]
        else:
            print("No neighborhoods supplied. Will use random neighbors.")
            self._find_closest_neighbors = lambda: np.tile(
                np.random.choice(self.N, size=(self.n_neighbors)), (self.N, 1))

        self.closest = self._find_closest_neighbors()

    def _maybe_update_ztree(self, iteration):
        try:
            if iteration % self.update_ztree_every == 0:
                self._make_z_tree()
        except AttributeError:
            # TODO: handle real neighborhoods.
            self._find_closest_neighbors = lambda: np.tile(
                np.random.choice(self.N, size=(self.n_neighbors)), (self.N, 1))
            return

    def _maybe_update_neighbors(self, iteration):
        if iteration % self.calc_closest_every == 0:
            self.closest = self._find_closest_neighbors()

    def _maybe_update_errors(self, iteration):
        if (iteration-1) % self.calc_dist_errors_every == 0:
            self.dist_errors = np.array(
                self.map(lambda i: self.calc_dist_errors(i), range(self.N)))

    def _calc_loss(self, iteration):
        losses = self._calc_losses(iteration)
        self.loss = np.sum(losses)
        self.losses_over_time.append(losses)
        return losses

    def _maybe_record_distances(self):
        if self.record_distances:
            distances = np.square(self.dist_errors)
            self.distances_over_time.append(np.mean(distances))

    def _maybe_log_status(self, iteration, losses):
        if iteration % self.verbosity == 0:
            log_string = "Iteration: {:d} Total Loss:{:.3f} Pred:{:.3f} ".format(iteration, np.sum(losses), losses[0])
            log_string += "Dist:{:.3f} l1:{:.3f} Phi_beta:{:.3f} ".format(losses[1], losses[2], losses[3])
            log_string += "Phi_u:{:.3f}, Beta_Scale:{:.3f} Patience: {:d}".format(losses[4], losses[5], self.patience)
            """
            if self.calc_com:
                com = np.linalg.norm(np.mean(beta_hat, axis=0) - self.init_beta[0, :], ord=2)
                mad = np.mean(np.array([
                    np.abs(beta_hat[i] - self.init_beta[0, :]) for i in range(N)]), axis=0)
                mad = np.linalg.norm(mad, ord=2) # Easier to read the logs if this is a single number, instead of per-feature.
                log_string += "\nCOM Divergence:{}\nMAD:{}".format(com, mad)
            """
            print(log_string, file=self.log)
            #print("phi_beta:{}\tphi_u:{}".format(phi_beta, phi_u), file=log)
            #print("beta_scale:{:.3f}".format(beta_scale), file=log)
            """
            print("Cosine Similarity between D_B and D_U:{:.3f}".format(
                np.mean(np.array([np.array([cosine_similarity([delta_beta(i,j)], [D[i, j]])[0] for j in range(N)])
                 for i in range(N)]))))
            """

    def _should_quit(self, iteration):
        if self.loss > 1e8:
            print("Diverged at iteration: {}".format(iteration))
            return True
        if self.loss > self.prev_loss - self.tol:
            self.patience -= 1
            if self.patience <= 0:
                print("Reached local minimum at iteration {:d}.".format(iteration))
                self.Q = self.Q_prev
                self.Z = self.Z_prev
                self._update_beta_hat()
                #self.nu = self.nu_prev
                self.phi_u = self.phi_u_prev
                return True

    def _update_opt_vars(self):
        self.lr *= self.lr_decay
        self.Q_prev = self.Q.copy()
        self.Z_prev = self.Z.copy()
        #self.nu_prev = self.nu
        self.phi_u_prev = self.phi_u.copy()
        self.prev_loss = self.loss


    def _reset_grads(self):
        self.grad_Z          = np.zeros_like(self.Z)
        self.grad_Q          = np.zeros_like(self.Q)
        self.grad_phi_u      = np.zeros_like(self.phi_u)
        #self.grad_nu         = np.zeros_like(self.nu)


    def _calc_prediction_grads(self):
        # Calculate Prediction Gradients
        grad_beta = np.array([
        	(self.f_prime(self.X[i], self.Y[i], self.beta_hat[i]) + \
            self.rho_beta_prime(self.beta_hat[i], i)
            )*1./(0.1+np.linalg.norm(self.beta_hat[i]-self.init_beta[i], ord=2))
            for i in range(self.N)])
        #print(self.Z, self.Q)
        if self.multitask:
            # grad_beta (N x P1 x P2)
            # self.Z (N x K)
            # self.Q (K x P1 x P2)
            self.grad_Q += np.tensordot(self.Z.T, grad_beta, axes=1) # K x P1 x P2
            self.grad_Z += np.tensordot(grad_beta, self.Q.swapaxes(0, 2).swapaxes(0, 1), axes=2) # N x K
        else:
            self.grad_Q += self.Z.T.dot(grad_beta) # K x P1 x P2
            self.grad_Z += grad_beta.dot(self.Q.T) # N x K


    def _calc_z_grad(self, i, de_i, closest_i):
        grad = (self.gamma / self.N) * 2 * np.mean([
            de_i[idx] * (self.Z[i] - self.Z[j]) for idx, j in enumerate(closest_i)], axis=0)
        return np.clip(grad, -1e0, 1e0)

    def _calc_phi_u_grad(self, i, de_i, closest_i):
        return (self.gamma / self.N)* 2 * np.mean([
            de_i[idx] * (-self.delta_U[i, j]) for idx, j in enumerate(closest_i)], axis=0)

    """
    def _calc_nu_grad(self, i, de_i, closest_i):
        return (self.gamma / self.N)* 2 * np.mean([
            de_i[idx] * np.linalg.norm(self.Z[i] - self.Z[j], ord=2)
            for idx, j in enumerate(closest_i)], axis=0)
    """


    def _calc_personalized_grads(self):
        """ Calculates the gradients for the DMR term."""
        def _calc_one(i): # Should help caching behavior.
            de_i = self.dist_errors[i] #np.sign(dist_errors[i]) # Derivative of squared term.
            closest_i = self.closest[i]
            self.grad_Z[i]       += self._calc_z_grad(i, de_i, closest_i)
            self.grad_phi_u   += self._calc_phi_u_grad(i, de_i, closest_i)
            #self.grad_nu      += self._calc_nu_grad(i, de_i, closest_i)

        self.map(_calc_one, list(range(self.N)))
        self.grad_phi_u += self.psi_u_prime(self.phi_u)
        self.grad_Z += self.Z
        self.grad_Q += self.Q
        #self.grad_nu    += self.psi_nu_prime(self.nu)


    def _calc_gradients(self):
        self._reset_grads()
        self._calc_prediction_grads()
        self._calc_personalized_grads()


    def _update_vars(self):
        if np.random.uniform() > 0.5:
            self.Z -= self.lr*self.grad_Z
        else:
            self.Q -= self.lr*self.grad_Q
        #phi_beta = soft_normalize(phi_beta - lr*grad_phi_beta)
        self.phi_u    = soft_normalize(self.phi_u - self.lr*self.grad_phi_u)
        #self.nu = self.nu#np.max([1e-5, self.nu - self.lr*self.grad_nu])
        self._update_beta_hat()


    def _single_restart(self, data, opt_params, log_params):
        #(X, Y, U, N, K, beta_hat, nu, phi_beta, phi_u, beta_prev, phi_beta_prev, phi_u, phi_u_prev)
        self._init_data_vars(data)
        self._init_opt_vars(opt_params)
        self.log = log_params["log"]
        self.verbosity = log_params["verbosity"]

        self.delta_z = lambda i, j: np.linalg.norm(self.Z[i] - self.Z[j], ord=2)
        self.dist_helper = lambda i, j: self.delta_z(i, j) - self.delta_U[i, j].dot(self.phi_u)
        self.calc_dist_errors = lambda i: np.array([
            self.dist_helper(i, j) for j in self.closest[i]])

        t = time.time()
        for iteration in range(1, self.max_iters+1):
            print("Iteration:{} of Max {}. Last Iteration Took {:.3f} seconds.".format(
                iteration, self.max_iters, time.time() - t), end='\r')
            t = time.time()
            self._maybe_update_ztree(iteration-1)
            self._maybe_update_neighbors(iteration-1)
            self._maybe_update_errors(iteration-1)
            losses = self._calc_loss(iteration-1)
            self._maybe_record_distances()
            self._maybe_log_status(iteration-1, losses)
            if self._should_quit(iteration):
                break
            self._update_opt_vars()
            self._calc_gradients()
            self._update_vars()
            self.log.flush()

        return #beta_hat, phi_beta, beta_scale, phi_u, loss, distances_over_time, losses_over_time

    def _maybe_make_delta_U(self, delta_U):
        if delta_U is None:
            print("Making Distances...")
            t = time.time()
            delta_U = self.make_covariate_distances(
                self.U, self.dU, self.K, self.N, should_normalize=True)
            print("Finished Making Distances. Took {:.3f} seconds.".format(time.time() - t))
        return delta_U


    def fit(self, X, Y, U, dU, delta_U=None, neighborhoods=None,
        init_lr=1e-3, lr_decay=1-1e-6, n_restarts=1,
        init_patience=10, max_iters=20000, tol=1e-3,
        verbosity=100, log_file=None, hierarchical=False,
        kd_leafsize=None, calc_neighbors=True,
        record_distances=False):
        """ Logistic Regression with Distance Matching.

            Parameters:

            X : Data
            Y : Data
        """
        N, K = self._check_shapes(X, Y, U, dU)
        self.U = U
        self.dU = dU
        self.K  = K
        self.N  = N
        delta_U = self._maybe_make_delta_U(delta_U)
        if kd_leafsize is None:
            kd_leafsize = self.n_neighbors

        self.best_loss = np.inf
        self.record_distances = record_distances
        if log_file is None:
            log_file ='{}/logs/distance_matching_{}.log'.format(
                self.log_dir, time.strftime("%Y_%m_%d-%H_%M_%S"))

        with open(log_file, 'a') as log:
            if hierarchical:
                self.f_single = self.f
                self.f_prime_single = self.f_prime
                self.f = lambda x, y, beta_hat: np.sum([
                    self.f_single(x[i], y[i], beta_hat) for i in range(len(x))])
                self.f_prime = lambda x, y, beta_hat: np.sum([
                    self.f_prime_single(x[i], y[i], beta_hat) for i in range(len(x))])
                # Condense X and Y into just two samples for each level.

                # Make hierarchy
                from sklearn.cluster import AgglomerativeClustering
                ag = AgglomerativeClustering(n_clusters=2,
                    compute_full_tree=False, affinity='precomputed', linkage='average')
                distance_mat = np.sum(delta_U, axis=2)
                print("Distance mat shape: {}".format(distance_mat.shape))
                """np.array([np.array([
                    delta_U[i, j].dot(np.ones_like(delta_U[i, j])) for j in range(len(delta_U))]) for i in range(len(delta_U))])"""

                def find_mean(U_mat):
                    from scipy.stats import mode
                    mean_vec = np.zeros((U_mat.shape[1]))
                    for j in range(U_mat.shape[1]):
                        try:
                            mean_vec[j] = np.mean(U_mat[:, j])
                        except ValueError:
                            mean_vec[j] = mode(U_mat[:, j])
                    return mean_vec

                def helper(idx, parent_beta, depth):
                    print("Depth={}".format(depth))
                    print(parent_beta.shape)
                    labels = ag.fit_predict(distance_mat[idx][:, idx])
                    X_idx = X[idx]
                    Y_idx = Y[idx]
                    U_idx = U[idx]
                    #X_clustered = np.zeros((2, max(len())))
                    X_clustered = np.array([X_idx[labels == 0, :], X_idx[labels == 1, :]])
                    Y_clustered = np.array([Y_idx[labels == 0, :], Y_idx[labels == 1, :]])
                    U_clustered = np.vstack((find_mean(U_idx[labels == 0, :]),
                        find_mean(U_idx[labels == 1, :])))
                    print(X_clustered)
                    print(U_clustered)
                    my_dist = np.array([dU[k](U_clustered[0][k], U_clustered[1][k]) for k in range(K)])
                    delta_U_clustered = np.array([
                        np.array([np.zeros_like(my_dist), my_dist]),
                        np.array([my_dist, np.zeros_like(my_dist)])])
                    print(delta_U_clustered.shape)
                    t = time.time()
                    self.init_beta = parent_beta#np.vstack((parent_beta, parent_beta))
                    (beta_hat, phi_beta, beta_scale,
                        phi_u, loss, distances_over_time,
                        losses_over_time) = self._single_restart(
                        X_clustered, Y_clustered, U_clustered,
                        delta_U_clustered, None, init_lr, lr_decay,
                        init_patience, max_iters, tol, verbosity, log)
                    idx0 = []
                    idx1 = []
                    for i, index in enumerate(idx):
                        if labels[i] == 0:
                            idx0.append(index)
                        else:
                            idx1
                    for i in idx0:
                        self.beta_hat[i] = beta_hat[0].copy()
                    for i in idx1:
                        self.beta_hat[i] = beta_hat[1].copy()
                    if len(idx0) > 1:
                        helper(idx0, beta_hat[0], depth+1)
                    if len(idx1) > 1:
                        helper(idx1, beta_hat[1], depth+1)

                    print("Took {:.3f} seconds.".format(time.time() - t))

                self.beta_hat = np.tile(self.init_beta, (X.shape[0]))
                print("beta_hat shape:{}".format(self.beta_hat.shape))
                helper(list(range(N)), self.beta_hat, 0)
            else:
                for restart in range(n_restarts):
                    t = time.time()
                    print("Restart {} of {}".format(restart+1, n_restarts))
                    self._single_restart({'X': X, 'Y': Y, 'U': U, 'delta_U': delta_U},
                        {'init_patience': init_patience, 'max_iters': max_iters,
                        'init_lr': init_lr, 'lr_decay': lr_decay, 'kd_leafsize': kd_leafsize,
                        'neighbors': neighborhoods, 'tol': tol, 'calc_neighbors': calc_neighbors},
                        {'verbosity': verbosity, 'log': log})
                    print("Took {:.3f} seconds.".format(time.time() - t))
                    if self.loss < self.best_loss:
                        print("** New best solution **")
                        self.best_loss = self.loss
                        self.best_Z     = self.Z.copy()
                        self.best_Q     = self.Q.copy()
                        self.best_beta_hat = self._update_beta_hat()
                        #self.best_beta_hat = self.Z.dot(self.best_Q)
                        #self.best_nu    = self.nu
                        self.best_phi_u = self.phi_u.copy()
                        self.best_distances_over_time = self.distances_over_time.copy()
                        self.best_losses_over_time = self.losses_over_time.copy()

        return self.best_Z, self.best_Q#, self.best_nu, self.best_phi_u, self.losses_over_time
