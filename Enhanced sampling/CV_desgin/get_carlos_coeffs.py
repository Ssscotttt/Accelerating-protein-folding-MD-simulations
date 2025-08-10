import numpy as np
from scipy.optimize import minimize

def optimize_weights(features_state1, features_state2):
    """
    Optimize the coefficient vector 'a' for all features by minimizing the scoring function.

    Parameters:
    - features_state1 (numpy array): Shape (N, T), where N is the number of features, and T is time series data from state 1.
    - features_state2 (numpy array): Shape (N, T), where N is the number of features, and T is time series data from state 2.

    Returns:
    - a_opt (numpy array): Optimized weights (size N) for all features.
    """
    n_feature = features_state1.shape[0]  # Number of features

    # Define the scoring function to minimize
    def psi(a):
        a = np.asarray(a).reshape(n_feature)
        print(a)
        """Scoring function for all features."""
        # Compute the weighted sum (combined CV)
        CV_1 = np.sum(a[:, np.newaxis] * features_state1, axis=0)  # Shape (T,)
        CV_2 = np.sum(a[:, np.newaxis] * features_state2, axis=0)  # Shape (T,)

        # Compute mean and standard deviation of the combined CV
        mean_CV_1 = np.mean(CV_1)
        mean_CV_2 = np.mean(CV_2)
        std_CV_1 = np.std(CV_1, ddof=1)
        std_CV_2 = np.std(CV_2, ddof=1)

        # Compute the discrimination term
        discrimination_term = - (mean_CV_1 - mean_CV_2) / (2 * (std_CV_1**2 + std_CV_2**2))

        # Compute the width balance term
        width_balance_term = max(std_CV_1, std_CV_2) / min(std_CV_1, std_CV_2)

        # Compute the regularization term (summed over all features)
        regularization_term = np.sum(a ** 2 * np.log(a ** 2 / (1 / n_feature)))

        return discrimination_term + width_balance_term + regularization_term

    # Initial guess: uniform weights
    a_init = np.ones(n_feature) / np.sqrt(n_feature)

    # Bounds: weights should be between 0 and 1
    bounds = [(0, 1) for _ in range(n_feature)]

    # Optimize 'a' for all features
    result = minimize(psi, a_init, bounds=bounds)

    return result.x  # Optimized weight vector

def read_features(colvar_file):
    data = np.genfromtxt(colvar_file)
    features = data[:, 1:]
    return features.T

features_1 = read_features("D:/Documents/Zhao/MRes/Enhanced_sampling/HemK/folded/COLVARS/COLVAR_native_rationals")
features_2 = read_features("D:/Documents/Zhao/MRes/Enhanced_sampling/HemK/unfolded/COLVARS/COLVAR_native_rationals")
a_opt = optimize_weights(features_1, features_2)

np.savetxt("D:/Documents/Zhao/MRes/Enhanced_sampling/HemK/a_opt.dat", a_opt, fmt="%.6f")

text = """
[0.         0.         0.         0.         0.76278143 1.
 0.90861979 1.         0.         0.69972982 1.         0.13555605
 0.56639488 0.         0.         0.2009496  0.58119548 0.00334785
 0.         0.         0.         0.87075657 0.         1.
 1.         1.         0.         0.         0.         0.
 1.         1.         1.         1.         1.         1.
 1.         1.         0.25260428 0.         1.         1.
 1.         1.         1.         0.32558093 1.         0.07616569
 0.02370898 0.         0.         0.         0.         0.14217831
 0.         0.         0.80478872 1.         1.         1.
 1.         1.         0.         0.71566356 0.         0.
 1.         1.         1.         1.         1.         0.40735783
 1.         1.         1.         1.         1.         1.
 1.         0.46499155 0.         1.         1.         1.
 1.         1.         1.         1.         0.29163049 0.01295243
 0.28052684 0.40928162 1.         1.         0.50063344 1.
 0.17009233 1.         1.         1.         1.         1.
 1.         1.         1.         0.49088639 0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.3895701  0.         0.
 0.         0.         0.         0.         0.74936549 0.
 0.         0.         0.         0.         1.         1.
 0.2142737  0.53490007 0.         1.         1.         1.
 0.         1.         1.         0.         1.         0.
 0.         0.22369088 0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.4093271  0.         0.20868351 0.00776387 0.39584579
 0.74549659 0.7937229  1.         0.         0.         0.04461315
 0.14141838 0.         0.         0.         0.         0.90291129
 0.87052209 0.         0.         0.         0.         0.
 0.         0.         0.         0.         1.         0.
 0.         0.28473343 0.         0.         0.         0.
 0.00568749 0.32706703 1.         0.         0.         0.33222539
 0.         0.         0.         0.79695125 0.         1.
 1.         1.         1.         1.         0.62588045 0.
 1.         1.         0.         0.         1.         0.
 0.         0.         0.         0.         0.         0.02363597
 0.01590172 0.88613627 0.79197248 0.65802434 0.4805432  0.50244976
 0.70411263 0.09988538 0.23692314 0.31381346 1.         0.77865353
 0.69535874 0.71042011 0.37052844 0.91168614 0.10490554 0.
 0.02630122 0.05084852 0.01251367 0.         0.         0.53378239
 0.31211402 0.         1.         1.         1.         0.
 0.         0.         1.         1.         0.48095123 0.
 0.         0.49648227 1.         1.         1.         0.50537169
 0.67274696 1.         0.38327309 0.05344431 0.39789429 1.
 1.         0.         0.         1.         1.         1.
 0.49467891 0.         0.17204277 0.         0.         0.91451073
 0.         0.         0.25065811 0.08155373 0.         0.
 0.48367793 0.         0.         0.         0.         0.15334946
 0.96250185 0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         1.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.02550026 0.         0.38213178 0.         0.18025931 0.0946215
 0.         0.         0.         0.         0.         0.24610673
 0.         0.         0.44356673 0.83749589 0.2504135  0.34046645
 0.63551452 1.         0.04904765 0.24737842 0.65059632 0.91168259
 0.97274059 1.         0.5512557  0.91826044 1.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.50228332 0.         0.         0.3541483
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.10179976 0.         0.39640215 0.67396412 1.
 0.12383902 0.61941913 1.         1.         0.32183957 0.77643948
 0.54348378 1.         1.         1.         1.         1.
 1.         0.65731646 0.50652917 0.5401183  0.45959036 0.
 0.         0.         1.         1.         1.         1.
 1.         1.         1.         1.         1.         1.
 1.         1.         1.         1.         1.         1.
 1.         1.         0.15599445 0.51192538 0.42029592 0.3863652
 0.30893025 0.4278934  0.4250118  0.44815772 0.         0.
 0.         0.79765629 0.08936775 0.15071157 0.24055881 0.
 0.93981848 0.06411044 0.19694818 1.         1.         1.
 1.         0.         1.         1.         1.         1.
 0.8282956  1.         1.         1.         0.78492798 1.
 0.34774205 0.70866276 1.         1.         1.         0.
 0.         0.27747044 1.         0.         1.         1.
 0.14746155 0.         0.20249326 0.31732833 0.         0.04585269
 0.         0.         0.         0.         0.33391489 0.
 0.         0.02382782 0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         1.         0.44452987 0.78188473 0.30695636 0.
 0.         0.         0.         0.         0.         0.
 0.15618737 0.         0.         0.         0.06648596 0.10987224
 0.58835314 0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.30655011 0.4269006  0.84170372
 0.         1.         0.57765218 1.         0.         1.
 0.63534861 0.77476047 0.86852777 1.         1.         0.80441692
 0.         0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.1314786
 0.         0.         0.         0.         0.         0.02530078
 0.02172675 0.45326813 0.54476205 0.4222389  0.44880078]
"""
import re
scalars = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", text)]
import pandas as pd
# Convert to DataFrame
df = pd.DataFrame(scalars, columns=["Scalars"])
df.to_csv("D:/Documents/Zhao/MRes/Enhanced_sampling/HemK/scalars.txt", index=False, header=False)