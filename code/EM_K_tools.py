from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import scipy.interpolate as si
import pandas as pd
import numpy as np
import os
from scipy.stats import norm
# from numpy import linalg as LA
# np.random.seed(1)

def initialize_means_stds(means, stds):
    """
    Turn iterables of per-state means and stds into
    (K, 1) column-vectors, whatever K happens to be.
    """
    means = np.asarray(means, dtype=float).reshape(-1, 1)
    stds  = np.asarray(stds,  dtype=float).reshape(-1, 1)

    if means.shape != stds.shape:
        raise ValueError("`means` and `stds` must have the same length")

    return means, stds, len(means)


def load_synthetic_exeperiment(method = 'Splines', contamenated=False, cont_percentage=0.2, samples = 500):
    
    """
    Generate synthetic data for a given set of parameters.

    Parameters
    ----------
    method : str, optional
        Method for basis time vector ('Splines' or 'Fourier').
    contamenated : bool, optional
        Whether to contaminate the data with outliers.
    cont_percentage : float, optional
        Percentage of contaminated data.
    samples : int, optional
        Number of samples to generate.

    Returns
    -------
    W_true : array-like
        Ground truth weights.
    MeansSigmas : array-like
        Ground truth means and standard deviations.
    K_true : int
        Ground truth number of states.
    e_data : array-like
        Observed smFRET values.
    t_data : array-like
        Time points corresponding to smFRET observations.
    z_data : array-like
        True state sequences.
    p : array-like
        Probability of each state at each time point.
    samples : int
        Number of samples.
    c : array-like
        Color for each state.
    """
    m1, m2, m3, m4 = 0.15, 0.25, 0.4, 0.6,
    std1, std2, std3, std4 = 0.01, 0.02, 0.03, 0.04
    MeansSigmas = [[m1,m2,m3,m4], [std1, std2, std3, std4]]
    K_true = 4
    dimt_true = 5

    T = 500
    W_true =  np.array([[ 0.,          0. ,         0.   ,       0. ,         0.        ],
    [-1.25,  8.51,  0.,          0.,          0.        ],
    [-1.45,  5.19,  0.,          0.,          0.        ],
    [-1.70,   3.98,  0.,          0.,          0.        ]])
    

    if contamenated:
        normal_samples = int((1-cont_percentage)*samples)
        contamenated_samples = int(cont_percentage*samples)
        t = np.random.uniform(0, T, normal_samples)
        e_data, t_data, z_data, p = generate_contamenated_data(W_true, MeansSigmas, normal_samples, t, dimt_true, T, method, contamenated_samples)
        c = cm.rainbow(np.linspace(0, 1, K_true+1))
    else:
        t = np.random.uniform(0, T, samples)
        e_data, t_data, z_data, p = generate_softmax_data(W_true, MeansSigmas, samples, t, dimt_true, T, method)
        c = cm.rainbow(np.linspace(0, 1, K_true))

    return W_true, MeansSigmas, K_true, e_data, t_data, z_data, p, samples, c

def get_prior_strength(p, samples):
    """
    Calculate the strength of the prior distribution (l) based on the given power (p) and number of samples.

    Parameters
    ----------
    p : float
        Power for the prior distribution strength.
    samples : int
        Number of samples.

    Returns
    -------
    l : float
        Strength of the prior distribution.
    """
    if p == 0:
        return 2 + 1e-5     # InvGamma's parameters "alpha" and "beta" must be positive. This yields to a very weak non-zero prior.
    l = samples**p
    return l

def vector_t(dimt, ti, T, AproximationMethod):
    """
    Generate basis time vector c(t) using the specified approximation method (B-splines or Fourier).

    Parameters
    ----------
    dimt : int
        Dimensionality of the basis time vector.
    ti : float
        Specific time point for which the vector is generated.
    T : float
        Total time period used for normalization.
    AproximationMethod : str
        Method for approximating the basis time vector ('Fourier' or 'Splines').

    Returns
    -------
    c_t : np.ndarray
        The generated basis time vector, transposed if using the 'Fourier' method.
    """

    if AproximationMethod == 'Fourier':
        c_t = np.ones((1,dimt))
        c_t[0][1] = ti/T
        for i in range(2,dimt,2):
            c_t[0][i] = np.cos(i*np.pi*ti/T)
        for i in range(3,dimt,2):
            c_t[0][i] = np.sin((i-1)*np.pi*ti/T)
        c_t = c_t.transpose()

    elif AproximationMethod == 'Splines':
        c_t = basis_splines(dimt, ti, T)

    else:
        raise ValueError('AproximationMethod variable can take 2 values ("Fourier" and "Splines")')
    return c_t

import pandas as pd

def read_real_data_xlsx(filename, *, sheet_name=0):
    """
    Read the 'Time' and 'FRET' columns from an .xlsx file produced by the
    spectro-scope and return them as 1-D NumPy arrays.
    
    Parameters
    ----------
    filename : str or path-like
        Path to the Excel workbook.
    sheet_name : str or int, optional
        Worksheet name or zero-based index (default 0 = first sheet).

    Returns
    -------
    t_data : numpy.ndarray
        1-D array of time values.
    e_data : numpy.ndarray
        1-D array of FRET values.

    Notes
    -----
    • Rows containing NaNs in either column are discarded.  
    • Requires `pip install pandas openpyxl` if not already available.
    """
    # Read only the two needed columns by header caption
    df = pd.read_excel(
        filename,
        sheet_name=sheet_name,
        usecols=["Time", "Fret"]
    ).dropna(how="any")

    t_data = df["Time"].to_numpy()
    e_data = df["Fret"].to_numpy()
    return t_data, e_data



def generate_contamenated_data(Wk, GroundTruth, samples, t, dimt, T, AproximationMethod, num_cont):
    """
    Generate a contaminated dataset with the specified number of contaminated samples.

    Parameters
    ----------
    Wk : array-like
        The matrix of weights for the Gaussian mixture model.
    GroundTruth : tuple
        The tuple of ground truth parameters (means, stds) for the Gaussian mixture model.
    samples : int
        The total number of samples to generate.
    t : array-like
        The time points at which to generate the samples.
    dimt : int
        The dimensionality of the basis time vector.
    T : float
        The total time period used for normalization.
    AproximationMethod : str
        The method for approximating the basis time vector ('Fourier' or 'Splines').
    num_cont : int
        The number of contaminated samples to generate.

    Returns
    -------
    e_data : list
        The generated contaminated dataset of smFRET values.
    t_data : list
        The generated contaminated dataset of time points.
    z_data : list
        The list of labels for the generated contaminated samples.
    p : list
        The list of probabilities for each Gaussian component at each time point.
    """
    uni = np.random.uniform(low=0.0, high=1.0, size=samples)
    K = Wk.shape[0]
    e_data,  z_data, t_data = [], [], []

    p = [[] for _ in range(K)]
    means = GroundTruth[0]
    stds = GroundTruth[1]

    for i in range(len(t)):
        e = [np.random.normal(means[j], stds[j], 1)[0] for j in range(K)]
        c_t = vector_t(dimt, t[i], 2*T, AproximationMethod)
        W = Wk@c_t
        W = [float(w) for w in W]
        h = sum(np.exp(-w) for w in W)
        for j in range(K):
            p[j].append(np.exp(-W[j])/h)
        z = 1
        while  uni[i] > sum(p[j][-1] for j in range(z)):
            z += 1
        e_data.append(e[z-1])
        t_data.append(t[i])
        z_data.append(z-1)

    if num_cont > 0:
        t_cont = np.random.choice(t, size=num_cont, replace=True)  # Random time points
        e_cont = np.random.uniform(0.1, 0.7, size=num_cont)        # Uniform noise
        z_cont = [-1] * num_cont                                   # Mark contaminated samples
        # p_cont = [1/K for _ in range(num_cont)]    

        e_data.extend(e_cont)
        t_data.extend(t_cont)
        z_data.extend(z_cont)
    return e_data, t_data, z_data, p

def distr_func(x, m, std):
    x , m, std = float(x), float(m), float(std)
    return norm.pdf(x, m, std)

def SaveContamenatedTruthParameters(path, t_data, e_data, z_data, c, W_true):
    """
    Save contaminated synthetic data and plot the FRET vs. Time scatter plot.

    This function saves the provided synthetic data (ground truth weights, smFRET values, 
    and time points) in the specified directory. It also generates a scatter plot of 
    smFRET values against time, highlighting contaminated data in red.

    Parameters
    ----------
    path : str
        Directory path to save the ground truth data and plot.
    t_data : list or array-like
        List of time points corresponding to the smFRET data.
    e_data : list or array-like
        List of smFRET values.
    z_data : list or array-like
        List of state labels for the data points. Contaminated data points are labeled with -1.
    c : list or array-like
        List of colors for each state, ignored in this function.
    W_true : array-like
        Ground truth matrix of weights for the Gaussian mixture model.

    Notes
    -----
    - The plot highlights contaminated data points in red.
    - The saved figure is named 'FRETvsTIME.png' and saved in the 'GroundTruth' subdirectory.
    - Ensure that the specified path exists or the function will create it.
    """

    palette = ['#88CCEE', '#117733', '#DDCC77', '#332288']
    plt.figure(figsize=(8, 6))
    K = W_true.shape[0]
    if not os.path.exists(path + 'GroundTruth'):
        os.makedirs(path + 'GroundTruth')

    # Save W_true, e_data, and t_data
    with open(path + 'GroundTruth/Wtrue.npy', 'wb') as f:
        np.save(f, W_true)
    with open(path + 'GroundTruth/e_data.npy', 'wb') as f:
        np.save(f, np.array(e_data))
    with open(path + 'GroundTruth/t_data.npy', 'wb') as f:
        np.save(f, np.array(t_data))

    # Prepare for plotting
    plt.xlabel('Time (s)')
    plt.ylabel("smFRET")
    plt.title("Synthetic Data with 10% Contamination")

    # Use the specified palette and add red for contamination
    color_list = palette[:K] + ['red']

    FretClasses = [[] for _ in range(K + 1)]      # Last list is for contamination
    Timeclasses = [[] for _ in range(K + 1)]

    for i in range(len(z_data)):
        z = z_data[i]
        idx = z if z >= 0 else K  # Contaminated data goes into index K
        FretClasses[idx].append(e_data[i])
        Timeclasses[idx].append(t_data[i])

    # Plot each class
    for i in range(K + 1):
        plt.scatter(Timeclasses[i], FretClasses[i], color=color_list[i], marker='o', s=4)

    # Save figure
    plt.xlim([-0.01,500])
    plt.grid(True)
    plt.savefig(path + 'GroundTruth/FRETvsTIME.png', dpi=300)
    # plt.clf()
    plt.close()


def generate_softmax_data(Wk, GroundTruth, samples, t, dimt, T, AproximationMethod):
    """
    Generate synthetic data using a softmax model with given parameters.

    Parameters
    ----------
    Wk : array-like
        Weight matrix for the Gaussian mixture model.
    GroundTruth : list of lists
        Ground truth means and standard deviations for Gaussian components.
    samples : int
        Number of samples to generate.
    t : array-like
        Array of time points for the samples.
    dimt : int
        Dimensionality of the basis time vector.
    T : float
        Total time period used for normalization.
    AproximationMethod : str
        Method for approximating the basis time vector ('Fourier' or 'Splines').

    Returns
    -------
    e_data : list
        Generated smFRET values.
    t_data : list
        Time points corresponding to each generated smFRET value.
    z_data : list
        State sequence for each generated sample.
    p : list of lists
        Probability of each state at each time point.
    """

    uni = np.random.uniform(low=0.0, high=1.0, size=samples)
    K = Wk.shape[0]
    e_data,  z_data, t_data = [], [], []

    p = [[] for _ in range(K)]
    means = GroundTruth[0]
    stds = GroundTruth[1]
    # print(np.random.rand())
    for i in range(len(t)):
        e = [np.random.normal(means[j], stds[j], 1)[0] for j in range(K)]
        c_t = vector_t(dimt, t[i], 2*T, AproximationMethod)
        W = Wk@c_t
        W = [float(w) for w in W]
        h = sum(np.exp(-w) for w in W)
        for j in range(K):
            p[j].append(np.exp(-W[j])/h)
        z = 1
        while  uni[i] > sum(p[j][-1] for j in range(z)):
            z += 1
        e_data.append(e[z-1])
        t_data.append(t[i])
        z_data.append(z-1)

    return e_data, t_data, z_data, p

def FrobDiff(West, Wtrue):
    """
    Compute the Frobenius norm of the difference between two matrices.

    Parameters
    ----------
    West : array-like
        Estimated matrix of weights.
    Wtrue : array-like
        Ground truth matrix of weights.

    Returns
    -------
    float
        Frobenius norm of the difference between West and Wtrue.
    """
    return LA.norm(Wtrue-West)

def SaveWestNorm(FrobNorms, iter, path):
    """
    Save a plot of the Frobenius norm of the difference between the estimated West matrix and the true West matrix over iterations.

    Parameters
    ----------
    FrobNorms : list
        List of Frobenius norms of the difference between the estimated West matrix and the true West matrix at each iteration.
    iter : int
        The number of iterations.
    path : str
        The path where the plot will be saved.

    Notes
    -----
    - The saved figure is named 'WestNorm_<iter>.png' and saved in the 'Norms/WestNorms' subdirectory.
    - Ensure that the specified path exists or the function will create it.
    """
    if not os.path.exists(path + 'Norms/WestNorms'):
        os.makedirs(path + 'Norms/WestNorms')

    plt.plot([i for i in range(iter)], FrobNorms, 'o-')
    plt.xlabel('Iterations')
    plt.ylabel('||$W_{true}$ - $W_{est}$||', rotation=90)
    plt.savefig(path + 'Norms/WestNorms/' + 'WestNorm_'+str(iter) + '.png')
    # plt.clf()
    plt.close()

def SaveMeansNorm(FrobNorms, iter, path):
    """
    Save a plot of the Frobenius norm of the difference between the estimated mean vectors and the true mean vectors over iterations.

    Parameters
    ----------
    FrobNorms : list
        List of Frobenius norms of the difference between the estimated mean vectors and the true mean vectors at each iteration.
    iter : int
        The number of iterations.
    path : str
        The path where the plot will be saved.

    Notes
    -----
    - The saved figure is named 'MeansNorm_<iter>.png' and saved in the 'Norms/MeansNorms' subdirectory.
    - Ensure that the specified path exists or the function will create it.
    """
    if not os.path.exists(path +  'Norms/MeansNorms'):
        os.makedirs(path + 'Norms/MeansNorms')

    plt.plot([i for i in range(iter)], FrobNorms, 'o-')
    plt.xlabel('Iterations')
    plt.ylabel('||$μ_{true}$ - $μ_{est}$||', rotation=90)
    plt.savefig(path + 'Norms/MeansNorms/' + 'MeansNorm_'+str(iter) + '.png')
    # plt.clf()
    plt.close()
    
def SaveStdsNorm(FrobNorms, iter, path):
    """
    Save a plot of the Frobenius norm of the difference between the estimated standard deviation vectors and the true standard deviation vectors over iterations.

    Parameters
    ----------
    FrobNorms : list
        List of Frobenius norms of the difference between the estimated standard deviation vectors and the true standard deviation vectors at each iteration.
    iter : int
        The number of iterations.
    path : str
        The path where the plot will be saved.

    Notes
    -----
    - The saved figure is named 'StdsNorm_<iter>.png' and saved in the 'Norms/StdsNorms' subdirectory.
    - Ensure that the specified path exists or the function will create it.
    """
    if not os.path.exists(path +  'Norms/StdsNorms'):
        os.makedirs(path + 'Norms/StdsNorms')

    plt.plot([i for i in range(iter)], FrobNorms, 'o-')
    plt.xlabel('Iterations')
    plt.ylabel('||$σ_{true}$ - $σ_{est}$||', rotation=90)
    plt.savefig(path + 'Norms/StdsNorms/' + 'StdsNorm_'+str(iter) + '.png')
    # plt.clf()
    plt.close()

def SaveQData(DataQ0, DataQ1, iter, path):
    """
    Save a plot of the negative log likelihood values Q0 and Q1 over iterations.

    This function generates and saves a line plot comparing the negative log likelihood 
    values Q0 and Q1 at each iteration. The plot is saved as a PNG file in the specified 
    directory.

    Parameters
    ----------
    DataQ0 : list
        List of negative log likelihood values Q0 at each iteration.
    DataQ1 : list
        List of negative log likelihood values Q1 at each iteration.
    iter : int
        The current number of iterations.
    path : str
        The directory path where the plot will be saved.

    Notes
    -----
    - The saved figure is named 'Qdata_<iter>.png' and saved in the 'Norms/Qdata' subdirectory.
    - Ensure that the specified path exists or the function will create it.
    """

    if not os.path.exists(path +  'Norms/Qdata'):
        os.makedirs(path + 'Norms/Qdata')
    
    plt.plot([i for i in range(iter)], DataQ0, 'ro-', label='$Q_{k}$')
    plt.plot([i for i in range(iter)], DataQ1, 'bo-', label='$Q_{k+1}$')
    plt.xlabel('Iterations')
    plt.legend()
    plt.ylabel('$Q_{k}$, $Q_{k+1}$', rotation=90)
    plt.savefig(path + 'Norms/Qdata/' + 'Qdata_'+str(iter) + '.png')
    # plt.clf()
    plt.close()

def SaveQdiff(Qdiff, iter, path):
    """
    Save a plot of the differences between the negative log likelihood values Q0 and Q1 over iterations.

    Parameters
    ----------
    Qdiff : list
        List of differences between the negative log likelihood values Q0 and Q1 at each iteration.
    iter : int
        The current number of iterations.
    path : str
        The directory path where the plot will be saved.

    Notes
    -----
    - The saved figure is named 'Qdiff_<iter>.png' and saved in the 'Norms/Qdiff' subdirectory.
    - Ensure that the specified path exists or the function will create it.
    """
    if not os.path.exists(path +  'Norms/Qdiff'):
        os.makedirs(path + 'Norms/Qdiff')
    
    plt.plot([i for i in range(iter)], Qdiff, 'o-')
    plt.xlabel('Iterations')
    plt.ylabel('||$Q_{k+1}$ - $Q_{k}$||', rotation=90)
    plt.savefig(path + 'Norms/Qdiff/' + 'Qdiff_'+str(iter) + '.png')
    # plt.clf()
    plt.close()

def SaveEstimatedMeansStdsFigures(iter, means, stds, GroundTruth, K, Ktrue, path, c):
    
    """
    Save a plot of the estimated means and standard deviations over current iterations.

    Parameters
    ----------
    iter : int
        The current number of iterations.
    means : list
        List of estimated mean values at each iteration.
    stds : list
        List of estimated standard deviation values at each iteration.
    GroundTruth : tuple
        Tuple of (Means, Stds) of the ground truth parameters.
    K : int
        The number of states.
    Ktrue : int
        The number of true states.
    path : str
        The directory path where the plot will be saved.
    c : list
        List of colors for each state.

    Notes
    -----
    - The saved figure is named 'means_stds_iter_<iter>.png' and saved in the 'ParametricEstimations/MeansStdsPlots' subdirectory.
    - Ensure that the specified path exists or the function will create it.
    """
    if not os.path.exists(path + 'ParametricEstimations/MeansStdsPlots'):
        os.makedirs(path + 'ParametricEstimations/MeansStdsPlots')
    if iter % 1 == 0:
        c = cm.rainbow(np.linspace(0, 1, K))
        # Plot means, stds graphs:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        for state in range(K):
            #Estimated:
            # print(stds , type(means))
            if isinstance(means, np.ndarray):  # Check if 'means' is a NumPy array
                means = means.tolist()
                stds = stds.tolist()

            mean_points = [means[i][state] for i in range(len(means))]
            std_points = [stds[i][state] for i in range(len(stds))]
        
            ax1.plot([i for i in range(0,iter)], mean_points,color=c[state], marker = 'o',linestyle='--',mfc='none', markersize=4)
            ax2.plot([i for i in range(0,iter)], std_points, color=c[state], marker = 'o',linestyle='--',mfc='none', markersize=4)

            #Ground Truth:
            if state < Ktrue:
                mean = GroundTruth[0][state]
                std = GroundTruth[1][state]
                ax1.plot([i for i in range(0,iter)], [mean for i in range(iter)],color=c[state], linestyle='-', markersize=4)
                ax2.plot([i for i in range(0,iter)], [std for i in range(iter)], color=c[state], linestyle='-', markersize=4)


        ax1.set_ylabel('$μ_k$')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('$σ_k$')
        plt.savefig(path + 'ParametricEstimations/MeansStdsPlots/' + 'means_stds_iter_'+ str(iter) +'.png')
        # plt.clf()
        plt.close()
        return c
def SaveGroundTruthParametersREAL(PathToSave, t_data, e_data):
    """
    Save real data and plot the FRET vs. Time scatter plot.

    This function saves the provided real data (time points and smFRET values) in the specified directory. It also generates a scatter plot of smFRET values against time.

    Parameters
    ----------
    PathToSave : str
        Directory path to save the ground truth data and plot.
    t_data : list or array-like
        List of time points corresponding to the smFRET data.
    e_data : list or array-like
        List of smFRET values.

    Notes
    -----
    - The saved figure is named 'FRETvsTIME.png' and saved in the 'GroundTruth' subdirectory.
    - Ensure that the specified path exists or the function will create it.
    """
    if not os.path.exists(PathToSave + 'GroundTruth/'):
        os.makedirs(PathToSave + 'GroundTruth/')
    
    plt.xlabel('Time (s)')
    plt.ylabel("apparent FRET")
    plt.title("Real Data")
    plt.figsize=(10, 6)
    plt.plot(t_data, e_data, 'ro', markersize=2)
    plt.grid(True)
    plt.xlim([-0.1,max(t_data)+0.1])
    plt.savefig(PathToSave + 'GroundTruth/' + 'FRETvsTIME' +'.png', dpi=300, format='png')
    # plt.clf()
    plt.close()


def SaveEstimatedMeansStdsREALFigures(iter, means, stds, K, path, c):
    """
    Save real data estimated parameters (means and stds) and plot the time evolution of means and stds.

    This function saves the estimated parameters (means and stds) of the real data in the specified directory. It also generates a plot of the time evolution of means and stds.

    Parameters
    ----------
    iter : int
        Current iteration number.
    means : array-like
        List of estimated means for each state.
    stds : array-like
        List of estimated standard deviations for each state.
    K : int
        Number of states.
    path : str
        Directory path to save the estimated parameters and plot.
    c : array-like
        List of colors for each state.

    Notes
    -----
    - The saved figure is named 'means_stds_iter_<iter>.png' and saved in the 'ParametricEstimations/MeansStdsPlots' subdirectory.
    - Ensure that the specified path exists or the function will create it.
    """
    if not os.path.exists(path + 'ParametricEstimations/MeansStdsPlots'):
        os.makedirs(path + 'ParametricEstimations/MeansStdsPlots')

    means = np.asarray(means)
    stds  = np.asarray(stds)

    if means.shape[0] != K and means.shape[1] == K:
        means = means.T
        stds  = stds.T

    if iter % 1 == 0:
        # Plot means, stds graphs:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        for state in range(K):
            mean_points = means[state,:]
            std_points = stds[state,:]
            ax1.plot([i for i in range(0,iter)], mean_points,color=c[state], marker = 'o',linestyle='--',mfc='none', markersize=4)
            ax2.plot([i for i in range(0,iter)], std_points, color=c[state], marker = 'o',linestyle='--',mfc='none', markersize=4)

        ax1.set_ylabel('$μ_k$')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('$σ_k$')
        plt.savefig(path + 'ParametricEstimations/MeansStdsPlots/' + 'means_stds_iter_'+ str(iter) +'.png')
        # plt.clf()
        plt.close()
        return c

def SaveEstimatedProbsFigures(Wk, t, T, dimt, iter, path, c, AproximationMethod):
    """
    Save estimated probabilities of each state at each time point and plot the time evolution of estimated probabilities.

    Parameters
    ----------
    Wk : array-like
        Estimated weights for each state.
    t : array-like
        Time points.
    T : float
        Time period.
    dimt : int
        Number of basis functions used to approximate the time evolution of the weights.
    iter : int
        Current iteration number.
    path : str
        Directory path to save the estimated probabilities and plot.
    c : array-like
        List of colors for each state.
    AproximationMethod : str
        Method used to approximate the time evolution of the weights.

    Notes
    -----
    - The saved figure is named 'probs_iter_<iter>.png' and saved in the 'ParametricEstimations/ProbsPlots' subdirectory.
    - Ensure that the specified path exists or the function will create it.
    - Returns the estimated probabilities of each state at each time point.
    """
    if not os.path.exists(path + 'ParametricEstimations/ProbsPlots'):
        os.makedirs(path + 'ParametricEstimations/ProbsPlots')
    K = Wk.shape[0]
    p = [[] for _ in range(K)]
    for i in range(len(t)):
        c_t = vector_t(dimt, t[i], 2*T, AproximationMethod)
        W = Wk@c_t
        W = [float(w) for w in W]
        h = sum(np.exp(-w) for w in W)

        for j in range(K):
            p[j].append(np.exp(-W[j])/h)

    t = np.asarray(t)
    order = np.argsort(t)          # indices that would sort t
    t_sorted = t[order]

    for i in range(K):
        p_i = np.asarray(p[i])[order]     # reorder the probabilities the same way
        plt.plot(t_sorted,
                p_i,
                color=c[i],
                linewidth=1.3)            # the smooth line
        plt.scatter(t, p[i],
                    color=c[i],
                    s=8,                  # keep markers for the raw points (optional)
                    zorder=3)

    plt.xlabel("Time (s)")
    plt.ylabel('Probability')
    plt.title("Estimated Probabilities")
    # plt.show()
    plt.savefig(path + 'ParametricEstimations/ProbsPlots/' 'probs_'+ str(iter) +'.png')
    # plt.clf()
    plt.close()

    return p

def SaveGroundTruthParameters(path, t_data, e_data, z_data, W_true, c, contaminated = False):
    """
    Save the ground truth parameters for a synthetic dataset and plot the smFRET values vs. time.

    Parameters
    ----------
    path : str
        Directory path to save the ground truth data and plot.
    t_data : list or array-like
        List of time points corresponding to the smFRET data.
    e_data : list or array-like
        List of smFRET values.
    z_data : list or array-like
        List of state labels for the data points.
    W_true : array-like
        Ground truth matrix of weights for the Gaussian mixture model.

    Notes
    -----
    - The saved figure is named 'FRETvsTIME.png' and saved in the 'GroundTruth' subdirectory.
    - Ensure that the specified path exists or the function will create it.
    """
    if contaminated:
        SaveContamenatedTruthParameters(path, t_data, e_data, z_data, c, W_true)
    else:
        K = W_true.shape[0]
        if not os.path.exists(path + 'GroundTruth'):
            os.makedirs(path + 'GroundTruth')

        with open(path + 'GroundTruth/' + 'Wtrue''.npy', 'wb') as f:                               # Save West
            np.save(f, W_true)
        with open(path + 'GroundTruth/' + 'e_data''.npy', 'wb') as f:                              # Save ei data
            np.save(f, np.array(e_data))
        with open(path + 'GroundTruth/' + 't_data''.npy', 'wb') as f:                              # Save ti data
            np.save(f, np.array(t_data))    

        plt.xlabel('Time (s)')
        plt.ylabel("apparent FRET")
        plt.title("Synthetic Data")
        FretClasses = [[] for _ in range(K)]
        Timeclasses = [[] for _ in range(K)]

        palette = ['#88CCEE', '#117733', '#DDCC77', '#332288']
        for i in range(len(z_data)):
            FretClasses[z_data[i]].append(e_data[i])
            Timeclasses[z_data[i]].append(t_data[i])

        for i in range(K):
            plt.scatter(Timeclasses[i], FretClasses[i], color = palette[i], marker = 'o', s=5)
        plt.savefig(path + 'GroundTruth/' + 'FRETvsTIME' +'.png')
        # plt.show()
        # plt.clf()
        plt.close()

def SaveSyntheticProbabilities(path, t, p, c):
    """
    Save and plot the synthetic probabilities of each state at different time points.

    This function generates a scatter plot of synthetic probabilities against time for
    each state using the provided probabilities and colors. The plot is then saved in
    the specified directory.

    Parameters
    ----------
    path : str
        Directory path to save the plot.
    t : list or array-like
        List of time points corresponding to the probabilities.
    p : list of lists
        List of probabilities for each state at each time point.
    c : list
        List of colors for each state.

    Notes
    -----
    - The saved figure is named 'TruthProbabilities.png' and saved in the 'GroundTruth' 
      subdirectory.
    - Ensure that the specified path exists or the function will create it.
    """

    if not os.path.exists(path + 'GroundTruth'):
        os.makedirs(path + 'GroundTruth')

    for i in range(len(p)):
        plt.scatter(t, p[i], color = c[i], marker = 'o', s=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Probabilites')
    plt.title('Generated Synthetic Probabilities')
    plt.savefig(path + 'GroundTruth/' + 'TruthProbabilites' +'.png')
    # plt.clf()
    plt.close()
    # plt.show()

def SaveParametricSpace(path, West, mk, stdk, iter):
    """
    Save parametric space at current iteration.

    This function saves the current parametric space of the weights, means, and standard deviations
    of the Gaussian mixture model in the specified directory.

    Parameters
    ----------
    path : str
        Directory path to save the parametric space.
    West : array-like
        Estimated weights for each state.
    mk : array-like
        Estimated means for each state.
    stdk : array-like
        Estimated standard deviations for each state.
    iter : int
        Current iteration number.

    Notes
    -----
    - The saved files are named 'West_<iter>.npy', 'mk_<iter>.npy', and 'stdk_<iter>.npy' and saved in the 'ParametricEstimations/West', 'ParametricEstimations/mk', and 'ParametricEstimations/stdk' subdirectories respectively.
    - Ensure that the specified path exists or the function will create it.
    """
    if not os.path.exists(path + 'ParametricEstimations/West'):
        os.makedirs(path + 'ParametricEstimations/West')

    if not os.path.exists(path + 'ParametricEstimations/mk'):
        os.makedirs(path + 'ParametricEstimations/mk')

    if not os.path.exists(path + 'ParametricEstimations/stdk'):
        os.makedirs(path + 'ParametricEstimations/stdk')

    with open(path + 'ParametricEstimations/West/' + 'West_' + str(iter) + '.npy', 'wb') as f:                              # Save West
        np.save(f, West)
    with open(path + 'ParametricEstimations/mk/' + 'mk_' + str(iter) + '.npy', 'wb') as f:                                  # Save mk
        np.save(f, mk)
    with open(path + 'ParametricEstimations/stdk/' + 'stdk_' + str(iter) + '.npy', 'wb') as f:                              # Save stdk
        np.save(f, stdk)
        

def generate_points(x):
    """
    Generate a list of random 2D points.

    This function generates a list of 2D points, with each point containing random integer
    coordinates. The first point is always [0, 0], and each subsequent point is generated
    randomly based on the specified parameter.

    Parameters
    ----------
    x : int
        The number of points to generate. Determines the range for random integer coordinates.
    
    Returns
    -------
    list of lists
        A list containing `x` number of 2D points, where each point is a list of two integers.
    """

    points = [[0, 0]]
    for i in range(1, x):
        points.append(np.random.randint(1, x*2+1, size=2).tolist())
    return points


def basis_splines(dimt, x0, T):
    """
    Generate basis time vector c(t) using B-splines.

    Parameters
    ----------
    dimt : int
        Dimensionality of the basis time vector.
    x0 : float
        Specific time point for which the vector is generated.
    T : float
        Total time period used for normalization.

    Returns
    -------
    c_t : np.ndarray
        The generated basis time vector, transposed if using the 'Fourier' method.
    """
    points = generate_points(dimt-2)
    points = np.array(points)
    t = np.linspace(0, T, len(points))
    x = points[:,0]
    y = points[:,1]

    x_tup = si.splrep(t, x, k=2)
    y_tup = si.splrep(t, y, k=2)
    
    basis_spline_values = [[1], [x0/T]]

    for i in range(len(points)):
        vec = np.zeros(len(points))
        vec[i] = 1.0
        x_list = list(x_tup)
        y_list = list(y_tup)
        x_list[1] = vec.tolist()
        y_list[1] = vec.tolist()
        basis_spline_values.append([si.splev(x0, x_list)])

    basis_spline_values = np.array(basis_spline_values)

    return basis_spline_values