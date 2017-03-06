import numpy as np
import scipy
from scipy.linalg import block_diag
if __name__ == '__main__':
    A = np.eye(4)*.2 + np.ones((4,4))*.2
    W = block_diag(A,A,A,A)
    vals, vecs = np.linalg.eig(W)
    print np.sort(vals)

