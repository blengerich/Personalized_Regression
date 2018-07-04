# Personalized Regression with Distance Matching Regularization
import numpy as np
np.set_printoptions(precision=4)
import time
from utils import *
from sklearn.preprocessing import normalize
from multiprocessing.pool import ThreadPool


class DistanceMatching():

    def __init__(self, init_beta,
                f, f_prime,
                gamma, n_neighbors, calc_closest_every,
                rho_beta, rho_beta_prime,
                init_phi_beta, psi_beta, psi_beta_prime,
                init_phi_u, psi_u, psi_u_prime,
                init_beta_scale, psi_beta_scale, psi_beta_scale_prime,
                intercept, log_dir="./logs", n_threads=1):

        """
        Create a new DistanceMatching object.

        Arguments
        ==========
        init_beta: numpy array of the initial model parameters. Should be of size N x P.
        f       : Python function for error of prediction error.
            Should take X^{(i)}, Y^{(i)}, beta^{(i)} and return a non-negative real value.
        f_prime : Python function for sub-gradient of prediction error.
            Should take X^{(i)}, Y^{(i)}, beta^{(i)} and return a sub-gradient vector of size P.
        gamma   : Hyperparameter for DMR strength.
        n_neighbors : Integer number of neighbors for each point.
        calc_closest_every: Integer number of iterations for which to re-calculate neighbors.
            Currently, neighbors are random so they should be computed relatively frequently.
        rho_beta : Python function for regularization of beta.
            Should take beta^{(i)} and return a non-negative real value.
        rho_beta_prime : Python function for sub-gradient of beta regularization.
            Should take beta^{(i)} and return a sub-gradient vector of size P.
        init_phi_beta : numpy array of the initial phi_beta vector. Should be of size P.
        psi_beta : Python function for regularization on phi_beta.
            Should take phi_beta and return a non-negative real value.
        psi_beta_prime : Python function for sub-gradient of phi_beta regularization.
            Should take phi_beta and return a sub-gradient vector of size P.
        init_phi_u : numpy array of the initial phi_u vector. Should be of size K.
        psi_u : Python function for regularization on phi_u.
            Should take phi_u and return a non-negative real value.
        psi_u_prime : Python function for sub-gradient of regularization of phi_u.
            Should take phi_u and return a sub-gradient vector of size K.
        init_beta_scale : Positive hyperparameter for the amount of personalization.
            Lower implies more personalization, as described in the paper.
        psi_beta_scale : Python function for regularization on beta_scale.
            Should take a postiive real value and return a non-negative real value.
        psi_beta_scale_prime: Python function for sub-gradient of beta scale regularization.
            Should take a positivie real value and return a sub-gradient.
        intercept : Boolean, whether to fit an intercept term.
        log_dir : string, directory to save output.
        n_threads : integer, max number of threads to use for multiprocessing.


        Returns
        ==========
        None
        """
        self.init_beta = init_beta
        self.f = f
        self.f_prime = f_prime
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.calc_closest_every = calc_closest_every
        self.rho_beta = rho_beta
        self.rho_beta_prime = rho_beta_prime
        self.init_phi_beta = init_phi_beta
        self.psi_beta = psi_beta
        self.psi_beta_prime = psi_beta_prime
        self.psi_beta_scale = psi_beta_scale
        self.psi_beta_scale_prime = psi_beta_scale_prime
        self.init_phi_u = init_phi_u
        self.psi_u = psi_u
        self.psi_u_prime = psi_u_prime
        self.init_beta_scale = init_beta_scale
        self.intercept = intercept
        self.log_dir = log_dir
        self.n_threads = n_threads
        if self.n_threads > 0:
            self.pool = ThreadPool(processes=self.n_threads)
            self.map = self.pool.map
        else:
            self.pool = None
            self.map = lambda x, y: list(map(x, y))

    def _check_shapes(self, X, Y, U=None, dU=None, delta_U=None):
        """ Does some basic checks on the shapes on the parameters. """
        N = X.shape[0]
        P = X.shape[1]
        if U:
            assert(U.shape[0] == N)
            K = U.shape[1]
        if dU:
            K = len(dU)
        if delta_U:
            assert(delta_U.shape[0] == N)
            assert(delta_U.shape[1] == N)
            K = delta_U.shape[2]
        return N, P, K

    def make_covariate_distances(self, U, dU, K, N, should_normalize=True, verbose=True):
        """ Make fixed pairwise distance matrix for co-variates. """
        t = time.time()
        if verbose:
            print("Making Co-Variate Distance Matrix of Size {}x{}x{}".format(N, N, K))
        D = np.zeros((N, N, K))
        get_dist = lambda i, j: np.array([dU[k](U[i, k], U[j, k]) for k in range(K)], dtype="float32")
        for i in range(1, N):
            if verbose:
                print("{}\t/{}".format(i, N), end='\r')
            D[i, 0:i, :] = self.map(lambda j: get_dist(i, j), range(i))
        for i in range(1, N):
            for j in range(i):
                D[j, i, :] = D[i, j, :] # could cut memory in half by only storing lists.
        if verbose:
            print("Finished making unnormalized version.")
        if should_normalize:
            normalized = np.array([normalize(D[:, :, k]) for k in range(K)])
            # Now the first axis references k. Move it to the back.
            normalized = np.swapaxes(normalized, 0, 1)
            D = np.swapaxes(normalized, 1, 2)
        if verbose:
            print("Finished normalizing.")
            print("Took {:.3f} seconds.".format(time.time() - t))
        return D

    def make_covariate_distance_function(self, U, dU, K):
        """ If N is large, it is more effecient to compute the covariate distances lazily. """
        func = lambda i,j: np.array([dU[k](U[i,k], U[j,k]) for k in range(K)])
        return func

    def _calc_personalized_reg_grad(self, phi_beta, phi_u, beta_hat, beta_scale,
                                    dist_errors, N, delta_U, delta_beta, closest):
        """ Calculates the gradients for the distance matching regularization.

            Arguments
            ==========

            phi_beta : numpy vector, current estimate of phi_beta
            phi_u    : numpy vector, current estimate of phi_u
            beta_hat : numpy matrix, current estimate of beta_hat
            beta_scale : float, current estimate of beta_scale
            dist_errors : list of lists of errrors.
            N           : integer number of samples.
            delta_U     : numpy matrix, static pairwise distance matrix.
            delta_beta  : Python function which calculates pairwise model distances.
            closest     : list of lists of closest indices.

            Returns
            =======

            grad_beta       : numpy matrix, sub-gradient wrt beta.
            grad_phi_beta   : numpy vector, sub-gradient wrt phi_beta.
            grad_phi_u      : numpy vector, sub-gradient wrt phi_u.
            grad_beta_scale : float, sub-gradient wrt beta_scale.
        """

        grad_phi_beta   = self.psi_beta_prime(phi_beta)
        grad_phi_u      = self.psi_u_prime(phi_u)
        grad_beta       = np.zeros_like(beta_hat)
        grad_beta_scale = self.psi_beta_scale_prime(beta_scale)

        def _calc_one_beta(i):
            return np.multiply(
                np.mean(np.array(
                    [dist_errors[i, idx]*np.sign(beta_hat[i] - beta_hat[j]) for idx, j in enumerate(closest[i])]), axis=0), phi_beta.T)

        def _calc_one_phi_beta(i):
            return np.mean(np.array([dist_errors[i, idx]*delta_beta(i, j) for idx, j in enumerate(closest[i])]), axis=0)

        def _calc_one_phi_u(i):
            return -np.mean(np.array([dist_errors[i, idx]*delta_U[i, j] for idx, j in enumerate(closest[i])]), axis=0)

        def _calc_one_beta_scale(i):
            return -np.mean(np.array([dist_errors[i, idx]*delta_beta(i, j) for idx, j in enumerate(closest[i])]), axis=0).dot(phi_beta)

        grad_beta       += self.gamma*np.array(self.map(_calc_one_beta, range(N)))
        grad_phi_beta   += self.gamma*np.mean(np.array(self.map(_calc_one_phi_beta, range(N))), axis=0)
        grad_phi_u      += self.gamma*np.mean(np.array(self.map(_calc_one_phi_u, range(N))), axis=0)
        grad_beta_scale += self.gamma*np.mean(np.array(self.map(_calc_one_beta_scale, range(N))), axis=0)

        return grad_beta, grad_phi_beta, grad_phi_u, grad_beta_scale


    def _single_restart(self, X, Y, delta_U, neighborhoods, init_lr, lr_decay,
                        init_patience, max_iters, tol, verbosity, log,
                        record_distances=False, calc_com=False):
        """ Execute a single restart of the optimization.

        Arguments
        =========
        X : numpy matrix of size NxP, design matrix
        Y : numpy vector of size Nx1, responses
        delta_U : numpy tensor of size NxNxK, constant covariate distances
        neighborhoods : list of list of neighbors
        init_lr : float, initial learning rate
        lr_decay : float, multiplicative factor by which to decay the learning rate.
        init_patience : integer, non-negative number of permitted iterations which
            don't decrease the loss functions.
        max_iters : integer, maximum number of iterations.
        tol : float, minimum amount by which the loss must decrease each iteration.
        verbosity: integer, every n iterations the current state will be logged.
        log : file pointer, log file
        record_distances : Boolean, whether to record pairwise distances during optimization.
        calc_com : Boolean, whether to calculate the center of mass (COM)
            deviation during optimiation.

        Returns
        ========

        beta_hat : numpy matrix of size NxP, estimate of model parameters.
        phi_beta : numpy vector of size P, estimate of phi_beta.
        beta_scale : float, estimate of personalization scaling factor.
        phi_u : numpy vector of size K, estimate of phi_u
        loss : float, final loss
        distances_over_time : list of distances during optimization
        losses_over_time : list of loss amounts during optimization
        """
        N, P, K    = self._check_shapes(X, Y, delta_U=delta_U)
        beta_hat   = self.init_beta.copy()
        beta_scale = self.init_beta_scale
        phi_beta   = self.init_phi_beta.copy()
        phi_u      = self.init_phi_u.copy()

        beta_prev     = self.init_beta.copy()
        phi_beta_prev = self.init_phi_beta.copy()
        phi_u_prev    = self.init_phi_u.copy()

        patience   = init_patience
        lr         = init_lr
        prev_loss  = np.inf
        distances_over_time = []
        losses_over_time = []

        if neighborhoods is None:
            print("No neighborhoods supplied. Will calculate neighbors randomly.")
            find_closest_neighbors = lambda phi_u: np.random.choice(N, size=(N, self.n_neighbors))
        else:
            print("Neighborhoods supplied. Will use those.")
            find_closest_neighbors = lambda phi_u: neighborhoods
        closest = find_closest_neighbors(phi_u)

        delta_beta = lambda i, j: np.abs(beta_hat[i] - beta_hat[j])
        dist_helper = lambda i, j: beta_scale*delta_beta(i, j).dot(phi_beta) - delta_U[i, j].dot(phi_u)
        calc_dist_errors = lambda i: np.array([dist_helper(i, j) for j in closest[i]])

        t = time.time()
        for iteration in range(1, max_iters+1):
            print("Iteration:{} of Max {}. Last Iteration Took {:.3f} seconds.".format(
                iteration, max_iters, time.time() - t), end='\r')
            t = time.time()

            if iteration % self.calc_closest_every == 0:
                closest = find_closest_neighbors(phi_u)

            loss1 = np.mean([self.f(X[i], Y[i], beta_hat[i].T) for i in range(N)])
            dist_errors = np.array(self.map(lambda i: calc_dist_errors(i), range(N)))
            loss2 = 0.5*self.gamma*np.mean(np.mean(np.square(dist_errors), axis=1))
            loss3 = np.mean([self.rho_beta(beta_hat[i]) for i in range(N)])
            loss4 = self.psi_beta(phi_beta)
            loss5 = self.psi_u(phi_u)
            loss6 = self.psi_beta_scale(beta_scale)
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            losses_over_time.append([loss1, loss2, loss3, loss4, loss5, loss6])

            if record_distances:
                distances = np.square(dist_errors)
                distances_over_time.append(np.mean(distances))
            if iteration % verbosity == 0:
                log_string = "Iteration: {:d} Total Loss:{:.3f} Pred:{:.3f} Dist:{:.3f} l1:{:.3f} Phi_beta:{:.3f} Phi_u:{:.3f}, Beta_Scale:{:.3f}".format(
                    iteration, loss, loss1, loss2, loss3, loss4, loss5, loss6)
                if calc_com:
                    com = np.linalg.norm(np.mean(beta_hat, axis=0) - self.init_beta[0, :], ord=2)
                    mad = np.mean(np.array([
                        np.abs(beta_hat[i] - self.init_beta[0, :]) for i in range(N)]), axis=0)
                    mad = np.linalg.norm(mad, ord=2) # Easier to read the logs if this is a single number, instead of per-feature.
                    log_string += "\nCOM Divergence:{}\nMAD:{}".format(com, mad)
                print(log_string, file=log)
            if loss > 1e8:
                print("Diverged at iteration: {}".format(iteration))
                break
            if loss > prev_loss - tol:
                patience -= 1
                if patience <= 0:
                    print("Reached local minimum at iteration {:d}.".format(iteration))
                    beta_hat = beta_prev
                    phi_beta = phi_beta_prev
                    phi_u = phi_u_prev
                    break

            lr *= lr_decay
            beta_prev = beta_hat.copy()
            phi_u_prev = phi_u.copy()
            phi_beta_prev = phi_beta.copy()
            prev_loss = loss

            # Calculate Gradients for Personalization Regularization
            grad_beta, grad_phi_beta, grad_phi_u, grad_beta_scale = self._calc_personalized_reg_grad(
                phi_beta, phi_u, beta_hat, beta_scale, dist_errors, N, delta_U, delta_beta, closest)

            # Calculate Gradients for Prediction
            for i in range(N):
                grad_beta[i] += self.f_prime(X[i], Y[i], beta_hat[i].T)
                grad_beta[i] += self.rho_beta_prime(beta_hat[i])
            beta_hat -= lr*grad_beta
            phi_beta = soft_normalize(phi_beta - lr*grad_phi_beta)
            if self.intercept:
                phi_beta[-1] = 0.  # intercept term does not count for personalization.
            phi_u = soft_normalize(phi_u - lr*grad_phi_u)
            beta_scale = np.max([1e-5, beta_scale - 1e-2*lr*grad_beta_scale])
            log.flush()

        return beta_hat, phi_beta, beta_scale, phi_u, loss, distances_over_time, losses_over_time

    def fit(self, X, Y, U, dU, delta_U=None, neighborhoods=None,
            init_lr=1e-3, lr_decay=1-1e-6, n_restarts=1,
            init_patience=10, max_iters=20000, tol=1e-3,
            verbosity=100, log_file=None):
        """ Fit the personalized model.

            Arguments
            =========
            X : numpy matrix of size NxP, design matrix
            Y : numpy vector of size Nx1, responses
            U : numpy matrix of size NxK, covariates
            dU: list of length K, each entry is a Python function for covariate-specific distance metric.
            delta_U: numpy tensor of size NxNxK, static covariate distances.
                If None, will be calculated before optimization starts.
            neighborhoods: list of list of neighbors.
                If None, neighborhoods will be generated during optimization.
            init_lr: float, learning rate.
            lr_decay: float, decay rate for learning rate.
            n_restarts : integer, number of restarts.
            init_patience: integer, number of iterations with non-decreasing loss before convergence is assumed.
            max_iters : integer, maximum number of iterations.
            tol : float, minimum decrease in loss.
            verbosity : integer, print output to log file every n iterations.
            log_file : str, filename of log file. If None, a new file will be created with the current datetime.


            Returns
            =======
            beta_hat : numpy matrix of size NxP, personalized model parameters
            phi_beta : numpy vector of size P, estimate of phi_beta
            phi_u    : numpy vector of size K, estimate of phi_u
            distances_over_time : list of pairwise distances during optimization
            losses_over_time : list of losses during optimization
        """

        N, P, K = self._check_shapes(X, Y, U, dU)
        if delta_U is None:
            print("Making Distances...")
            t = time.time()
            delta_U = self.make_covariate_distances(U, dU, K, N, should_normalize=True)
            print("Finished Making Distances. Took {:.3f} seconds.".format(time.time() - t))
        best_loss = np.inf

        if log_file is None:
            log_file = "{}/distance_matching_{}.log".format(
                self.log_dir, time.strftime("%Y_%m_%d-%H_%M_%S"))

        with open(log_file, 'a') as log:
            for restart in range(n_restarts):
                t = time.time()
                print("Restart {} of {}".format(restart+1, n_restarts))
                (beta_hat, phi_beta, beta_scale,
                    phi_u, loss, distances_over_time, losses_over_time) = self._single_restart(
                    X, Y, U, delta_U, neighborhoods, init_lr, lr_decay,
                    init_patience, max_iters, tol, verbosity, log)
                print("Took {:.3f} seconds.".format(time.time() - t))
                if loss < best_loss:
                    best_loss = loss
                    print("** New best solution **")
                    self.loss = loss
                    self.beta_hat = beta_hat.copy()
                    self.phi_beta = phi_beta.copy()
                    self.phi_u = phi_u.copy()
                    self.distances_over_time = distances_over_time.copy()
                    self.losses_over_time = losses_over_time.copy()

        return self.beta_hat, self.phi_beta, self.phi_u, self.distances_over_time, self.losses_over_time