from scipy.optimize import minimize
from utils import randomise
import numpy as np

def initialize_W(q:int):

    W = np.full(shape=[q**2,q**2,q**2,q**2], fill_value=np.nan)
    W[0,0,0,0] = 1
    for i in  range(q**2):
        for j in range(q**2):
            if i != 0 or j != 0: 
                W[i,j,0,0] = 0
                W[i,0,j,0] = 0
                W[0,0,i,j] = 0
                W[0,i,0,j] = 0 
    return W

def repack_W(indices,X,W):

    #right now this is tailored for qubits
    for i in range(len(X)):
        index = indices[i]
        e = X[i]
        W[index[0],index[1],index[2],index[3]] = e

def find_W_constrained(q):

    Num_unfixed = q**8 - (4*q**4 - 4*q**2 + 1)
    X_0 = np.random.rand(Num_unfixed)
    R = np.random.rand(Num_unfixed)

    def unitarity(indices,X,W):

        repack_W(indices,X,W)
        IdId = np.einsum("ac,bd->abcd", np.identity(q**2), np.identity(q**2))
        return np.linalg.norm(np.einsum('abcd,efcd -> abef', W, W) - IdId)

    def dual_unitarity(indices,X,W):
        
        repack_W(indices,X,W)
        IdId = np.einsum("ac,bd->abcd", np.identity(q**2), np.identity(q**2))
        return np.linalg.norm(np.einsum('fbea,fdec->abcd', W, W)-IdId)

    def soliton(indices,X,W):

        repack_W(indices,X,W)
        I, Z = np.zeros(q**2), np.zeros(q**2)
        I[0], Z[1] = 1, 1
        IZ = np.einsum('a,b->ab',I,Z)
        ZI = np.einsum('a,b->ab',Z,I)
        return np.linalg.norm(np.einsum('abcd,cd->ab',W,ZI)-IZ) + np.linalg.norm(np.einsum('abcd,cd->ab',W,IZ)-ZI)

    W = initialize_W(q)
    indices = np.argwhere(np.isnan(W))

    cons = [{'type':'eq', 'fun': lambda z:  unitarity(indices, z, W)},
            {'type':'eq', 'fun': lambda z: dual_unitarity(indices, z, W)},
            {'type':'eq', 'fun': lambda z: soliton(indices, z, W)}]

    result = minimize(
        lambda z: randomise(z, R), x0 = X_0, method='SLSQP', \
             constraints=cons, options={'maxiter': 1000}
                     )
        
    return result.x