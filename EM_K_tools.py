import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import random
from scipy.stats import norm
import scipy.interpolate as si
# from sympy import *
import pandas as pd
import math as m
from numpy import linalg as LA
from matplotlib.pyplot import cm
from sklearn.metrics import mean_squared_error
np.random.seed(42)

def load_synthetic_exeperiment():
    m1, m2, m3, m4 = 0.15, 0.25, 0.4, 0.6,
    std1, std2, std3, std4 = 0.01, 0.02, 0.03, 0.04
    GroundTruth = [[m1,m2,m3,m4], [std1, std2, std3, std4]]
    K_true = 4
    dimt_true = 5
    samples = 500
    T = 500
    t = np.random.uniform(0, T, samples)
    method = 'Splines'
    W_true =  np.array([[ 0.,          0. ,         0.   ,       0. ,         0.        ],
    [-1.25161591,  8.51426105,  0.,          0.,          0.        ],
    [-1.45448931,  5.19878006,  0.,          0.,          0.        ],
    [-1.7006334,   3.98199831,  0.,          0.,          0.        ]])
    
    e_data, t_data, z_data, p = generate_softmax_data(W_true, GroundTruth, samples, t, dimt_true, T, method)
    c = cm.rainbow(np.linspace(0, 1, K_true))

    return W_true, GroundTruth, K_true, e_data, t_data, z_data, p, samples, c


def get_t(dimt, ti, T, AproximationMethod):
    if AproximationMethod == 'Fourier':
        t_hat = np.ones((1,dimt))
        t_hat[0][1] = ti/T
        for i in range(2,dimt,2):
            t_hat[0][i] = np.cos(i*np.pi*ti/T)
        for i in range(3,dimt,2):
            t_hat[0][i] = np.sin((i-1)*np.pi*ti/T)
        t_hat = t_hat.transpose()

    elif AproximationMethod == 'Splines':
        t_hat = basis_splines(dimt, ti, T)

    else:
        raise ValueError('AproximationMethod variable can take 2 values ("Fourier" and "Splines")')
    return t_hat

def distr_func(x, m, std):
    x , m, std = float(x), float(m), float(std)
    return norm.pdf(x, m, std)

def read_real_data():
    xls = pd.ExcelFile('/Users/andrew_kwf/Downloads/apo_MBP_folding_Fret_vs_time.xlsx')
    df1 = pd.read_excel(xls, 'fretvstime', header=None)
    df1 = np.array(df1)
    e_data, t_data = [], []

    for i in range(df1.shape[0]):
        if m.isnan(df1[i][1]) or m.isnan(df1[i][0]):
            continue 
        t_data.append(df1[i][0])
        e_data.append(df1[i][1])

    return e_data, t_data

def generate_softmax_data(Wk, GroundTruth, samples, t, dimt, T, AproximationMethod):
    # np.random.seed(42)
    uni = np.random.uniform(low=0.0, high=1.0, size=samples)
    K = Wk.shape[0]
    e_data,  z_data, t_data = [], [], []

    p = [[] for _ in range(K)]
    means = GroundTruth[0]
    stds = GroundTruth[1]
    # print(np.random.rand())
    for i in range(len(t)):
        e = [np.random.normal(means[j], stds[j], 1)[0] for j in range(K)]
        t_hat = get_t(dimt, t[i], 2*T, AproximationMethod)
        W = Wk@t_hat
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

    print('DATA COUNTERS: (z=1, z=2, z=3, z=4, z=5) = (', z_data.count(0),z_data.count(1),z_data.count(2),z_data.count(3),z_data.count(4), ')')
    return e_data, t_data, z_data, p

def FrobDiff(West, Wtrue):
    return LA.norm(Wtrue-West)
    # return mean_squared_error(Wtrue,West)

def SaveWestNorm(FrobNorms, iter, path):
    if not os.path.exists(path + 'Norms/WestNorms'):
        os.makedirs(path + 'Norms/WestNorms')

    plt.plot([i for i in range(iter)], FrobNorms, 'o-')
    plt.xlabel('Iterations')
    plt.ylabel('||$W_{true}$ - $W_{est}$||', rotation=90)
    plt.savefig(path + 'Norms/WestNorms/' + 'WestNorm_'+str(iter) + '.png')
    plt.clf()

def SaveMeansNorm(FrobNorms, iter, path):
    if not os.path.exists(path +  'Norms/MeansNorms'):
        os.makedirs(path + 'Norms/MeansNorms')

    plt.plot([i for i in range(iter)], FrobNorms, 'o-')
    plt.xlabel('Iterations')
    plt.ylabel('||$μ_{true}$ - $μ_{est}$||', rotation=90)
    plt.savefig(path + 'Norms/MeansNorms/' + 'MeansNorm_'+str(iter) + '.png')
    plt.clf()
    
def SaveStdsNorm(FrobNorms, iter, path):
    if not os.path.exists(path +  'Norms/StdsNorms'):
        os.makedirs(path + 'Norms/StdsNorms')

    plt.plot([i for i in range(iter)], FrobNorms, 'o-')
    plt.xlabel('Iterations')
    plt.ylabel('||$σ_{true}$ - $σ_{est}$||', rotation=90)
    plt.savefig(path + 'Norms/StdsNorms/' + 'StdsNorm_'+str(iter) + '.png')
    plt.clf()

def SaveQData(DataQ0, DataQ1, iter, path):
    if not os.path.exists(path +  'Norms/Qdata'):
        os.makedirs(path + 'Norms/Qdata')
    
    plt.plot([i for i in range(iter)], DataQ0, 'ro-', label='$Q_{k}$')
    plt.plot([i for i in range(iter)], DataQ1, 'bo-', label='$Q_{k+1}$')
    plt.xlabel('Iterations')
    plt.legend()
    plt.ylabel('$Q_{k}$, $Q_{k+1}$', rotation=90)
    plt.savefig(path + 'Norms/Qdata/' + 'Qdata_'+str(iter) + '.png')
    plt.clf()

def SaveQdiff(Qdiff, iter, path):

    if not os.path.exists(path +  'Norms/Qdiff'):
        os.makedirs(path + 'Norms/Qdiff')
    
    plt.plot([i for i in range(iter)], Qdiff, 'o-')
    plt.xlabel('Iterations')
    plt.ylabel('||$Q_{k+1}$ - $Q_{k}$||', rotation=90)
    plt.savefig(path + 'Norms/Qdiff/' + 'Qdiff_'+str(iter) + '.png')
    plt.clf()

def SaveEstimatedMeansStdsFigures(iter, means, stds, GroundTruth, K, Ktrue, path, c):
    if not os.path.exists(path + 'ParametricEstimations/MeansStdsPlots'):
        os.makedirs(path + 'ParametricEstimations/MeansStdsPlots')
    if iter % 1 == 0:
        c = cm.rainbow(np.linspace(0, 1, K))
        # K = max(K1,Ktrue)
        # print(K)
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


        # ax1.set_title('Means',fontsize=8, y=1.0, pad=-14)
        # ax1.set_xlabel('Iteration')
        ax1.set_ylabel('$μ_k$')
        # ax2.set_title('Standard deviation',fontsize=8, y=1.0, pad=-14)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('$σ_k$')
        plt.savefig(path + 'ParametricEstimations/MeansStdsPlots/' + 'means_stds_iter_'+ str(iter) +'.png')
        plt.clf()
        return c

def SaveEstimatedMeansStdsREALFigures(iter, means, stds, K, path, c):
    if not os.path.exists(path + 'ParametricEstimations/MeansStdsPlots'):
        os.makedirs(path + 'ParametricEstimations/MeansStdsPlots')

    if iter % 1 == 0:
        # Plot means, stds graphs:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        for state in range(K):
            #Estimated:
            mean_points = means[state,:]
            std_points = stds[state,:]
            ax1.plot([i for i in range(0,iter)], mean_points,color=c[state], marker = 'o',linestyle='--',mfc='none', markersize=4)
            ax2.plot([i for i in range(0,iter)], std_points, color=c[state], marker = 'o',linestyle='--',mfc='none', markersize=4)

        # ax1.set_title('Means',fontsize=8, y=1.0, pad=-14)
        # ax1.set_xlabel('Iteration')
        ax1.set_ylabel('$μ_k$')
        # ax2.set_title('Standard deviation',fontsize=8, y=1.0, pad=-14)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('$σ_k$')
        plt.savefig(path + 'ParametricEstimations/MeansStdsPlots/' + 'means_stds_iter_'+ str(iter) +'.png')
        plt.clf()
        return c

def SaveEstimatedProbsFigures(Wk, t, T, dimt, iter, path, c, AproximationMethod):
    if not os.path.exists(path + 'ParametricEstimations/ProbsPlots'):
        os.makedirs(path + 'ParametricEstimations/ProbsPlots')
    K = Wk.shape[0]
    p = [[] for _ in range(K)]
    for i in range(len(t)):
        t_hat = get_t(dimt, t[i], 2*T, AproximationMethod)
        W = Wk@t_hat
        W = [float(w) for w in W]
        h = sum(np.exp(-w) for w in W)

        for j in range(K):
            p[j].append(np.exp(-W[j])/h)

    for i in range(K):
        plt.scatter(t,p[i],color = c[i], marker = 'o', s=2)

    plt.xlabel("Time (s)")
    plt.ylabel('Probability')
    plt.title("Estimated Probabilities")
    # plt.show()
    plt.savefig(path + 'ParametricEstimations/ProbsPlots/' 'probs_'+ str(iter) +'.png')
    plt.clf()

    return p

def CreateWk4Synthetic(K, dimt):
    W = np.zeros((K,dimt))
    
    # e1 = np.random.normal(0, 1, 1)[0]
    # rng = random.uniform(-0.1,0.1)
    for k in range(1,K):
        e1 = np.random.normal(0, 1, 1)[0]
        # rng = np.random.uniform(-1,1)
        rng = np.random.uniform(-1, 1)
        W[k][0] = 1/(k+1)*(10*rng)
        W[k][1] = 1/(k+1)*(10+5*e1)
        # W[k][0] = 1/(k+1)*(1+0.1*e1)
        # W[k][1] = 1/(k+1)*rng
    return W

def SaveGroundTruthParameters(path, t_data, e_data, z_data, c, W_true):
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
    plt.ylabel("smFret")
    plt.title("Synthetic Data")
    FretClasses = [[] for _ in range(K)]
    Timeclasses = [[] for _ in range(K)]
    for i in range(len(z_data)):
        FretClasses[z_data[i]].append(e_data[i])
        Timeclasses[z_data[i]].append(t_data[i])

    for i in range(K):
        plt.scatter(Timeclasses[i], FretClasses[i], color = c[i], marker = 'o', s=2)
    plt.savefig(path + 'GroundTruth/' + 'FRETvsTIME' +'.png')
    # plt.show()
    plt.clf()

def SaveSyntheticProbabilities(path, t, p, c):
    if not os.path.exists(path + 'GroundTruth'):
        os.makedirs(path + 'GroundTruth')

    for i in range(len(p)):
        plt.scatter(t, p[i], color = c[i], marker = 'o', s=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Probabilites')
    plt.title('Generated Synthetic Probabilities')
    plt.savefig(path + 'GroundTruth/' + 'TruthProbabilites' +'.png')
    plt.clf()
    # plt.show()

def SaveParametricSpace(path, West, mk, stdk, iter):
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
        
def ReadApoData():
    path = '/Users/andrew_kwf/Desktop/ApoRealData.csv'
    data = np.array(pd.read_csv(path))
    e_data, t_data = [], []
    for i in range(data.shape[0]):

        t_data.append(data[i][0])
        e_data.append(data[i][1])
    return e_data, t_data

def generate_points(x):
    points = [[0, 0]]
    # np.random.seed(42)
    for i in range(1, x):
        points.append(np.random.randint(1, x*2+1, size=2).tolist())
        # points.append([random.randint(1, x*2), random.randint(1, x*2)])
    return points

def getDerivativeOft_hat(dimt, ti, T, AproximationMethod):
    if AproximationMethod == 'Fourier':
        t_hat = np.zeros((1,dimt))
        t_hat[0][1] = 1/T
        for i in range(2,dimt,2):
            t_hat[0][i] = -np.sin(i*np.pi*ti/T) * (i*np.pi/T)
        for i in range(3,dimt,2):
            t_hat[0][i] = np.cos((i-1)*np.pi*ti/T) * (i*np.pi/T)

    elif AproximationMethod == 'Splines':
        points = generate_points(dimt-2)
        points = np.array(points)
        t = np.linspace(0, T, len(points))
        x = points[:,0]
        y = points[:,1]

        x_tup = si.splrep(t, x, k=2)
        y_tup = si.splrep(t, y, k=2)
        
        t_hat = [[0], [1/T]]  # Derivative of constant and linear terms

        for i in range(len(points)):
            vec = np.zeros(len(points))
            vec[i] = 1.0
            x_list = list(x_tup)
            y_list = list(y_tup)
            x_list[1] = vec.tolist()
            y_list[1] = vec.tolist()
            t_hat.append([si.splev(ti, x_list, der=1)])  # Compute derivative

        t_hat = np.array(t_hat).T
    return t_hat.transpose()

def ComputePenaltyScalar(Wk, t_data, T, Lambda, AproximationMethod):
    K, dimt = Wk.shape
    c_primes = np.array([getDerivativeOft_hat(dimt, t_data[i], 2*T, AproximationMethod) for i in range(len(t_data))])
    Products = np.array([float(c_primes[i].T @ Wk.T @ Wk @ c_primes[i]) for i in range(len(c_primes))])
    time_values = np.linspace(0, T, num=len(Products))
    Penal = np.trapz(Products, time_values)
        
    return Lambda*Penal

# TO TEST WITH DERIVATIVES:
def basis_splines(dimt, x0, T):
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

# #1st:
# # Wtrue =  [[ 0.          0.          0.          0.          0.        ]
# #  [ 1.56869599  4.71468097  0.          0.          0.        ]
# #  [-0.43637438  3.38728447  0.          0.          0.        ]
# #  [-0.84160249  2.69081503  0.          0.          0.        ]]


# #2nd
# # Wtrue =  [[ 0.          0.          0.          0.          0.        ]
# #  [-1.25161591  8.51426105  0.          0.          0.        ]
# #  [-1.45448931  5.19878006  0.          0.          0.        ]
# #  [-1.7006334   3.98199831  0.          0.          0.        ]]

# #3rd
# # Wtrue =  [[ 0.          0.          0.          0.          0.        ]
# #  [-1.36401242  8.56094469  0.          0.          0.        ]
# #  [-2.01079252  5.43279727  0.          0.          0.        ]
# #  [ 1.52779078  3.72894595  0.          0.          0.        ]]

# #4th
# # Wtrue =  [[ 0.          0.          0.          0.          0.        ]
# #  [-0.44597541  7.99785346  0.          0.          0.        ]
# #  [-2.05854073  4.46536958  0.          0.          0.        ]
# #  [-1.64818622  3.80579235  0.          0.          0.        ]]