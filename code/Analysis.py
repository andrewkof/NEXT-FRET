from scipy.optimize import minimize
import numpy as np
from .EM_K_tools import *
import autograd.numpy as np  # Use autograd's NumPy
from autograd import grad
from autograd import hessian
from scipy.stats import invgamma


class tvGMM:
    def __init__(self, e, t, K, dimt, method, initial_means, initial_stds):
        """
        Time-Varying Gaussian Mixture Model (tvGMM) class.
        
        Parameters
        ----------
        e : array-like
            Observed smFRET values.
        t : array-like
            Time points corresponding to smFRET observations.
        K : int
            Number of states (components in the mixture model).
        dimt : int
            Dimensionality of the basis time vector.
        method : str
            Method for basis time vector ('Splines' or 'Fourier').
        initial_means : array-like
            Initial estimates for the means of the Gaussian components.
        initial_stds : array-like
            Initial estimates for the standard deviations of the Gaussian components.
        """
        self.e = np.array(e)
        self.t = np.array(t)
        self.K = K
        self.dimt = dimt
        self.method = method
        self.T = max(t)
        self.Mk = np.array(initial_means, dtype=float)
        self.STDk = np.array(initial_stds, dtype=float)
        self.Wk = np.zeros((K, dimt))
        self.responsibilities = None
        # Flag to use prior knowledge or not
        self.UsePrior = False
        # Prior knowledge
        self.PriorKnowledge = None
        # Ground truth parameters for synthetic data
        self.GroundTruth = None
        self.K_true = None
        # Flag for synthetic data run
        self.SyntheticRun = False
        # Color for each state
        self.c = cm.rainbow(np.linspace(0, 1, self.K))
        # Store means and standard deviations across iterations
        self.StoreMeans = [self.Mk.flatten().tolist()]
        self.StoreSTDs = [self.STDk.flatten().tolist()]

    def enable_prior(self, PriorKnowledge):
        """
        Enable prior knowledge usage.

        Parameters
        ----------
        PriorKnowledge : list
            List of prior knowledge parameters.
        """
        self.PriorKnowledge = PriorKnowledge
        self.UsePrior = True
    
    def setup_synthetic(self, MeansStds_true, K_true):
        """
        Set up synthetic experiment's parameters.

        Parameters
        ----------
        MeansStds_true : array-like
            Ground truth means and standard deviations.
        K_true : int
            Ground truth number of states.
        """
        self.GroundTruth = MeansStds_true
        self.K_true = K_true
        # Flag to indicate whether the model is running on synthetic data or not
        self.SyntheticRun = True

    def Q_omega(self, Wk_1, mk_1, stdk_1):
        """
        Expectation of the complete-data log likelihood  Q function.

        Parameters
        ----------
        Wk_1 : array-like
            Weights at previous iteration.
        mk_1 : array-like
            Means at previous iteration.
        stdk_1 : array-like
            Standard deviations at previous iteration.

        Returns
        -------
        Q : float
            Q-function value.
        """
        K = self.Wk.shape[0]
        state_k_1 = self.responsibilities

        # Compute emission probabilities
        PeGivenz = [[] for _ in range(K)]
        for i in range(len(self.e)):
            for j in range(K):
                PeGivenz[j].append(distr_func(self.e[i], mk_1[j], stdk_1[j]))

        # Compute transition probabilities
        PzGivent = [[] for _ in range(K)]
        for i in range(len(self.e)):
            t_hat = vector_t(self.dimt, self.t[i], 2 * self.T, self.method)
            W = Wk_1 @ t_hat
            W = [float(w) for w in W]
            h = sum(np.exp(-w) for w in W)
            for j in range(K):
                PzGivent[j].append(np.exp(-W[j]) / h)

        # Compute Q-function
        r = sum(
            [sum(state_k_1[j][i] * np.log(PeGivenz[j][i] * PzGivent[j][i] + 10**(-10)) for j in range(K)) for i in range(len(self.t))]
        )

        if self.UsePrior:
            # Compute prior likelihood
            prior_likelihood = float(self.compute_prior_likelihood(mk_1, stdk_1))
            # Return total Q-function value
            return r + prior_likelihood

        return r
    
    def compute_prior_likelihood(self, mk_1, stdk_1):
        """
        Compute the prior likelihood contribution to the Q-function.

        Parameters
        ----------
        mk_1 : array-like
            Means at the previous iteration.
        stdk_1 : array-like
            Standard deviations at the previous iteration.

        Returns
        -------
        float
            Total prior likelihood contribution.
        """
        # Initialize prior terms
        prior_mu_terms, prior_std_terms = 0, 0
        # Unpack prior knowledge
        priors_mu, priors_std = self.PriorKnowledge

        # Calculate prior terms for means
        for PriorMu in priors_mu:
            state, mu_0, lambda_mu = PriorMu
            m_k = mk_1[state - 1]
            # Accumulate log probability of the mean
            prior_mu_terms += np.log(distr_func(m_k, mu_0, 1 / lambda_mu) + 1e-5)

        # Calculate prior terms for standard deviations
        for PriorStd in priors_std:
            state, s_0, lambda_s = PriorStd
            s_k = stdk_1[state - 1]
            alpha = 0.5 * lambda_s - 1
            beta = 0.5 * lambda_s * (s_0 ** 2)
            # Accumulate log probability of the standard deviation
            prior_std_terms += np.log(invgamma.pdf(s_k**2, alpha, scale=beta) + 1e-5)
        
        # Return the total prior likelihood
        return prior_mu_terms + prior_std_terms

    def estimate_mean_std(self, probs, prior=None):
        """
        Estimate new means and stds from the probabilities and data.

        Parameters
        ----------
        probs : array-like
            Probabilities of each observation belonging to each state.
        prior : tuple, optional
            Prior knowledge for one state. If given, it should be a tuple of (mean, lambda_mean, std, lambda_std).

        Returns
        -------
        mean_est : float
            Estimated mean of the state.
        std_est : float
            Estimated standard deviation of the state.
        """
        # Compute the numerator for the mean estimate
        num_mean = np.dot(probs, self.e)
        # Compute the denominator for the mean estimate
        denom = np.sum(probs) + (prior[1] if prior else 0)
        # Compute the estimated mean
        mean_est = (num_mean + (prior[1] * prior[0] if prior else 0)) / denom if denom != 0 else 0
        
        # Compute the numerator for the std estimate
        num_std = np.dot(probs, (self.e - mean_est) ** 2) + (prior[3] * (prior[2] ** 2) if prior else 0)
        # Compute the denominator for the std estimate
        denom = np.sum(probs) + (prior[3] if prior else 0)
        # Compute the estimated std
        std_est = np.sqrt(num_std / denom) if denom != 0 else 1e-5
        
        return mean_est, std_est

    def vector_h_numpy(self, w):
        """
        Compute the negative log likelihood for the vector h.

        Parameters
        ----------
        w : array-like
            Weight matrix to be reshaped and used for computation.

        Returns
        -------
        float
            The negative log likelihood result.
        """
        # Reshape w into (K, dimt) and ensure the first row is fixed to zero
        w = w.reshape(self.K, self.dimt)
        w = np.vstack([np.zeros((1, self.dimt)), w[1:, :]])

        # Precompute t_hat, a matrix with shape (N, dimt)
        t_hat = np.array([vector_t(self.dimt, ti, 2*self.T, self.method).ravel() for ti in self.t])

        # Compute matrix product X and its negation
        X = t_hat @ w.T
        Xneg = -X

        # Calculate softmax denominator with shape (N, 1)
        p = np.exp(Xneg).sum(axis=1, keepdims=True)

        # Compute log softmax
        log_softmax = Xneg - np.log(p)

        # Transpose responsibilities matrix for efficient computation
        terms_np_T = np.array(self.responsibilities).T  # shape (N, K)

        # Compute the sum of the product of terms_np_T and log_softmax
        result = np.sum(terms_np_T * log_softmax)

        # Return the negative result to maximize the likelihood
        return -result
        
    def M_Step(self):
        """Maximization Step

        Updates Means, Stds and Matrix W using the computed responsibilities
        and prior knowledge if available.
        """
        mk_1, stdk_1, prior_index = [], [], 0
        # Update Means and Stds:
        for i in range(self.K):
            if self.UsePrior and self.PriorKnowledge and (i + 1) in [p[0] for p in self.PriorKnowledge[0]]:
                # Use prior knowledge to update mean and std
                prior_mu = self.PriorKnowledge[0][prior_index][1:]
                prior_std = self.PriorKnowledge[1][prior_index][1:]
                prior_index += 1
                mean_est, std_est = self.estimate_mean_std(self.responsibilities[i], prior=prior_mu + prior_std)
            else:
                # Update mean and std without prior knowledge
                mean_est, std_est = self.estimate_mean_std(self.responsibilities[i])
            mk_1.append(mean_est)
            stdk_1.append(std_est)
        
        # Update Matrix W:
        # Use the Newton-CG optimization method to minimize the negative log likelihood
        # The optimization method requires the gradient and Hessian of the function
        optimized_W = minimize(
            fun = lambda w: self.vector_h_numpy(w),
            x0 = self.Wk.flatten(),
            method = 'Newton-CG',
            jac = grad(self.vector_h_numpy),
            hess = hessian(self.vector_h_numpy),
        ).x.reshape(self.K, self.dimt)
        
        return optimized_W, np.array(mk_1), np.array(stdk_1)

    def E_Step(self):
        """Expectation Step

        Computes the responsibilities matrix p(z|ei,ti) for all states
        """
        t_hat = [vector_t(self.dimt, self.t[i], 2*self.T, self.method) for i in range(len(self.t))]                 # Matrix like, shape=(dimt,len(ei)). Contains t vectors for all ti.
        Wi = [self.Wk@t_hat[i] for i in range(len(self.t))]                                                      # Matrix like, shape=(K,len(ei)). Contains all matrix multiplications for all ti. 

        wi, terms, pi_denom = [], [], []
        for state in range(self.K):
            wi.append([Wi[i][state][0] for i in range(len(Wi))])                                                    # Get state's row from Wi
            pi = [np.exp(-wi[state][i])*distr_func(self.e[i], self.Mk[state], self.STDk[state]) for i in range(len(wi[state]))] # Calculate Psoftmax for each state. (without denominator)
            terms.append(pi)

        for i in range(len(self.e)):                                                                                # Calculate summarization of probs to normalize later.        
            denom = 0
            for state in range(self.K):
                denom += terms[state][i]
            pi_denom.append(denom)

        for state in range(self.K):                                                  # Normilize probabilites.
            for i in range(len(terms[state])):
                terms[state][i] /= pi_denom[i]
        terms = np.array(terms)  
        return terms

    def EM_algorithm(self, max_iters, path):
        """
        Expectation-Maximization Algorithm to find optimal parametric space (Means, Stds and matrix W).

        Parameters
        ----------
        max_iters : int
            Maximum number of iterations.
        path : str
            Path to save the results.
        """
        DataQ0, DataQ1, Qdiff, self.path, iter = [], [], [], path, 1

        while iter <= max_iters:

            # Save estimated means and stds
            if self.SyntheticRun:
                SaveEstimatedMeansStdsFigures(iter, self.StoreMeans, self.StoreSTDs, self.GroundTruth, self.K, self.K_true, path, self.c)
            else:
                SaveEstimatedMeansStdsREALFigures(iter, self.StoreMeans, self.StoreSTDs, self.K, path, self.c)
            
            # Save estimated probabilities
            SaveEstimatedProbsFigures(self.Wk, self.t, self.T, self.dimt, iter, path, self.c, self.method)
            SaveParametricSpace(path, self.Wk, self.Mk, self.STDk, iter)

            # Perform E-Step
            self.responsibilities = self.E_Step()               

            # Store previous step's means and stds
            self.MuSigmaPrevStep = [self.Mk, self.STDk]

            # Perform M-Step
            Wk_1, mk_1, stdk_1 = self.M_Step()                 

            # Calculate negative log likelihood
            Q1 = self.Q_omega(Wk_1, mk_1, stdk_1)               
            Q0 = self.Q_omega(self.Wk, self.Mk, self.STDk)

            # Store differences in negative log likelihood
            if iter>1:
                Qdiff.append(Q1 - Q0)
                DataQ0.append(Q0)
                DataQ1.append(Q1)
                SaveQData(DataQ0, DataQ1, iter-1, path)
                SaveQdiff(Qdiff, iter-1, path)

            # Print the difference in negative log likelihood
            print()
            print('Iteration: ', iter)
            print('Q1 - Q0 = ', Q1-Q0)

            # Check convergence
            if Q1 - Q0 <= 1e-4 and iter > 1:
                print('Converged to a local maxima at iteration', iter)
                break
        

            # Reshape means and stds for next iteration
            mk_1 = mk_1.reshape(self.K, 1)
            stdk_1 = stdk_1.reshape(self.K, 1)

            # Append current iteration's means and stds to the storage arrays
            self.StoreMeans.append(mk_1.flatten())  
            self.StoreSTDs.append(stdk_1.flatten())
            
            # Increment iteration counter
            iter+=1
            # Update current step's means and stds
            self.Wk, self.Mk, self.STDk = Wk_1, mk_1, stdk_1

        # Transform storage arrays to numpy arrays
        self.StoreMeans = np.array(self.StoreMeans).T           # Shape (K, Iterations)
        self.StoreSTDs = np.array(self.StoreSTDs).T             # Shape (K, Iterations)

        # Return the optimal parametric space
        return self.Wk, self.StoreMeans, self.StoreSTDs